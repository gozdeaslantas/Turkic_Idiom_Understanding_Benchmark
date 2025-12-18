# Turkic Idiom Span Detection

A unified, reproducible benchmark and experimental framework for **idiom span detection**
across **five Turkic languages**:

**Turkish (TR), Azerbaijani (AZ), Turkmen (TK), Gagauz (GA), and Uzbek (UZ)**.

This repository provides an end-to-end pipeline for building morphology-aware idiom
span datasets, training token-level NER models, and evaluating **zero-shot** and
**few-shot cross-lingual transfer**.

---

## 📌 Task Definition

**Task:** Idiom span detection  
**Formulation:** Token-level sequence labeling (NER)  
**Tagging scheme:** BIO

| Label     | Description                          |
|-----------|--------------------------------------|
| `O`       | Non-idiom token                      |
| `B-IDIOM` | Beginning of idiomatic expression   |
| `I-IDIOM` | Inside idiomatic expression          |

Given a sentence containing an idiom, the goal is to identify the **exact token span**
corresponding to the idiomatic expression.

---

## 🌍 Languages

| Code | Language     |
|------|--------------|
| TR   | Turkish      |
| AZ   | Azerbaijani  |
| TK   | Turkmen      |
| GA   | Gagauz       |
| UZ   | Uzbek        |

---

## 🧠 Models

The framework supports any HuggingFace token-classification model.
Default experiments include:

- XLM-R (`xlm-roberta-base`)
- mBERT (`bert-base-multilingual-cased`)
- BERTurk
- ELECTRA-tr
- DistilBERT-tr
- ConvBERT-tr
- Turkish-e5-large

All models are defined **declaratively** in `configs/default.yaml`.
