from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import numpy as np

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

from config import LABEL_LIST, LABEL2ID, ID2LABEL, MAX_LENGTH
from utils.seed import set_seed
from modeling.metrics import compute_metrics
from modeling.callbacks import EpochLogger
from modeling.tokenize import TokenAligner

@dataclass
class FewShotRunner:
    langs: List[str]
    shots: List[Optional[int]]          # None => full
    epochs: int = 5
    num_proc: int = 4
    max_length: int = MAX_LENGTH

    @staticmethod
    def choose_lr(n_shot: Optional[int]) -> float:
        # Colab mantığıyla uyumlu: low-shot düşük LR, full daha yüksek
        if n_shot in [10, 30, 50]:
            return 2e-5
        return 5e-5

    def _sample(self, ds_split, lang: str, n: Optional[int], seed: int = 42):
        ds_lang = ds_split.filter(lambda ex: ex["lang"] == lang)
        if len(ds_lang) == 0:
            return ds_lang
        if n is None or n >= len(ds_lang):
            return ds_lang
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds_lang), size=n, replace=False)
        return ds_lang.select(idx.tolist())

    def _tokenize(self, ds_split, tokenizer):
        if len(ds_split) == 0:
            return ds_split
        aligner = TokenAligner(tokenizer, max_length=self.max_length)
        tok = ds_split.map(
            aligner,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=ds_split.column_names,
        )
        return tok.with_format("torch")

    def run(
        self,
        tag: str,
        base_ckpt_dir: str,              # trainer_tr.state.best_model_checkpoint
        ds: DatasetDict,
        tokenizer_name: str,
        out_dir: str | Path,
        seed: int = 42,
    ) -> List[dict]:
        set_seed(seed)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        results: List[dict] = []

        for L in self.langs:
            val_raw = ds["validation"].filter(lambda ex: ex["lang"] == L)
            test_raw = ds["test"].filter(lambda ex: ex["lang"] == L)
            if len(test_raw) == 0:
                continue

            for n_shot in self.shots:
                shot_name = "full" if n_shot is None else f"{n_shot}shot"
                fs_dir = out_dir / f"{L}_{shot_name}"
                fs_dir.mkdir(parents=True, exist_ok=True)

                train_raw = self._sample(ds["train"], L, n_shot, seed=seed)

                train_ds = self._tokenize(train_raw, tokenizer)
                val_ds = self._tokenize(val_raw, tokenizer) if len(val_raw) else None
                test_ds = self._tokenize(test_raw, tokenizer)

                model = AutoModelForTokenClassification.from_pretrained(
                    base_ckpt_dir,
                    num_labels=len(LABEL_LIST),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID
                )

                log_path = fs_dir / "training_history.json"
                args = TrainingArguments(
                    output_dir=str(fs_dir),
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    num_train_epochs=self.epochs,
                    learning_rate=self.choose_lr(n_shot),
                    weight_decay=0.01,
                    load_best_model_at_end=True,
                    metric_for_best_model="f1",
                    greater_is_better=True,
                    logging_steps=20,
                    save_total_limit=1
                )

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    callbacks=[EpochLogger(str(log_path))]
                )

                trainer.train()
                metrics = trainer.evaluate(test_ds)
                metrics.update({"lang": L, "shot": shot_name})

                (fs_dir / "test_metrics.json").write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                results.append(metrics)

        (out_dir / "few_shot_metrics.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return results
