from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json

from datasets import DatasetDict
from transformers import AutoTokenizer

from config import MAX_LENGTH
from modeling.tokenize import TokenAligner

@dataclass
class ZeroShotEvaluator:
    langs: List[str]
    num_proc: int = 4
    max_length: int = MAX_LENGTH

    def _tokenize_lang(self, ds_split, tokenizer, lang: str):
        ds_lang = ds_split.filter(lambda ex: ex["lang"] == lang)
        if len(ds_lang) == 0:
            return ds_lang
        aligner = TokenAligner(tokenizer, max_length=self.max_length)
        tok = ds_lang.map(
            aligner,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=ds_split.column_names,
        )
        return tok.with_format("torch")

    def evaluate(
        self,
        tag: str,
        trainer,
        ds: DatasetDict,
        tokenizer_name: str,
        out_dir: str | Path,
    ) -> Dict[str, dict]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        results: Dict[str, dict] = {}
        for L in self.langs:
            test_L = self._tokenize_lang(ds["test"], tokenizer, L)
            if len(test_L) == 0:
                continue
            metrics = trainer.evaluate(test_L)
            results[L] = metrics

        out_path = out_dir / "zero_shot_metrics.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return results
