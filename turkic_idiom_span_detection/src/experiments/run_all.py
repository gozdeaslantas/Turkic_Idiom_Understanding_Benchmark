from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from utils.config_loader import Config
from utils.seed import set_seed
from config import LANGS
from data.build_dataset import DatasetBuilder
from data.split import split_by_idiom_id   # aşağıda veriyorum
from experiments.train_tr import TurkishTrainer
from experiments.eval_zero_shot import ZeroShotEvaluator
from experiments.few_shot import FewShotRunner
from modeling.hub import HubPusher, HubVersion


def ensure_dirs(*paths: str | Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def build_hf_dataset_from_ner_df(ner_df: pd.DataFrame) -> DatasetDict:
    train_df = ner_df[ner_df["split"] == "train"].drop(columns=["split"])
    val_df   = ner_df[ner_df["split"] == "validation"].drop(columns=["split"])
    test_df  = ner_df[ner_df["split"] == "test"].drop(columns=["split"])

    return DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })


def main(cfg_path: str):
    cfg = Config.load(cfg_path)
    set_seed(cfg.get("seed", 42))

    # paths
    base_out = Path(cfg.dig("runs", "base_out", default="runs"))
    results_out = Path(cfg.dig("runs", "results_out", default=str(base_out / "results")))
    ensure_dirs(base_out, results_out)

    # load csv
    csv_path = Path(cfg.dig("data", "csv_path"))
    df = pd.read_csv(csv_path)
    df["idiom_id"] = df.index

    # build NER
    builder = DatasetBuilder()
    ner_df = builder.build(df)

    # split by idiom_id
    ner_df = split_by_idiom_id(ner_df, test_size=0.15, val_size=0.10, seed=cfg.get("seed", 42))

    # save labeled CSV
    labeled_out = Path(cfg.dig("data", "output_labeled_csv", default="data/processed/ner_labeled_multi_turkic.csv"))
    labeled_out.parent.mkdir(parents=True, exist_ok=True)
    ner_df.to_csv(labeled_out, index=False)

    # HF DatasetDict for training
    ds = build_hf_dataset_from_ner_df(ner_df)

    # Evaluators
    zero_eval = ZeroShotEvaluator(
        langs=cfg.dig("zero_shot", "langs", default=["AZ", "TK", "GA", "UZ"]),
        num_proc=cfg.get("num_proc", 4),
        max_length=cfg.get("max_length", 256),
    )
    few_runner = FewShotRunner(
        langs=cfg.dig("few_shot", "langs", default=["AZ", "TK", "GA", "UZ"]),
        shots=cfg.dig("few_shot", "shots", default=[10, 30, 50, None]),
        epochs=cfg.dig("few_shot", "epochs", default=5),
        num_proc=cfg.get("num_proc", 4),
        max_length=cfg.get("max_length", 256),
    )

    # HF push (models)
    hf_do_push = cfg.dig("hf", "do_push_models", default=False)
    hf_private = cfg.dig("hf", "private", default=True)
    org = cfg.dig("hf", "org_or_user", default="Gozde")
    repo_prefix = cfg.dig("hf", "repo_prefix", default="idiom-span")
    commit_msg = cfg.dig("hf", "commit_message", default="Add model")

    ver_cfg = cfg.dig("hf", "versioning", default={}) or {}
    version = HubVersion(
        enabled=bool(ver_cfg.get("enabled", False)),
        scheme=str(ver_cfg.get("scheme", "date")),
        semver=str(ver_cfg.get("semver", "v1.0.0")),
        extra=str(ver_cfg.get("extra", "")),
    )
    pusher = HubPusher(
        org_or_user=org,
        repo_prefix=repo_prefix,
        private=hf_private,
        commit_message=commit_msg,
        version=version,
    )

    models: Dict[str, str] = cfg.dig("models", default={})
    all_results: Dict[str, Any] = {"tr": {}, "zero_shot": {}, "few_shot": {}}

    for tag, model_name in models.items():
        run_dir = base_out / f"{tag}_tr"
        run_dir.mkdir(parents=True, exist_ok=True)

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --- TR TRAIN ---
        tr_trainer = TurkishTrainer(model_name=model_name, out_dir=str(run_dir))
        # IMPORTANT: train_tr.py sınıfın train() metodu tokenize edilmiş ds bekliyor.
        # Biz burada ds["train"/"validation"] içinden TR filtreli tokenize'i üretmeliyiz.
        from datasets import Dataset
        from modeling.tokenize import TokenAligner

        def tokenize_lang(split, lang: str):
            ds_lang = split.filter(lambda ex: ex["lang"] == lang)
            if len(ds_lang) == 0:
                return ds_lang
            aligner = TokenAligner(tokenizer, max_length=cfg.get("max_length", 256))
            tok = ds_lang.map(
                aligner,
                batched=True,
                num_proc=cfg.get("num_proc", 4),
                remove_columns=split.column_names,
            )
            return tok.with_format("torch")

        train_tr = tokenize_lang(ds["train"], "TR")
        val_tr = tokenize_lang(ds["validation"], "TR")
        test_tr = tokenize_lang(ds["test"], "TR")

        trainer = tr_trainer.train(train_tr, val_tr)
        tr_metrics = trainer.evaluate(test_tr)
        all_results["tr"][tag] = tr_metrics

        # --- PUSH TR BEST CHECKPOINT ---
        if hf_do_push:
            meta = {
                "tag": tag,
                "base_model": model_name,
                "task": "Turkic idiom span detection (TR train)",
                "split": "TR",
            }
            pusher.push_checkpoint(
                tag=tag,
                run_dir=run_dir,
