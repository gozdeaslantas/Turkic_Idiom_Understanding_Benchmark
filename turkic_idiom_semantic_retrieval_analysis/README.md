# Turkic Idiom Semantic Retrieval & Analysis

A unified experimental framework for **semantic retrieval and cross-lingual alignment of
idiomatic expressions** across **Turkic languages**.

This repository investigates **how well transformer-based language models capture the
figurative meaning of idioms**, using embedding-based retrieval, fine-tuning, and
cross-lingual evaluation.

---

## 📌 Research Scope

Idioms are **non-compositional multi-word expressions**, posing a challenge for
distributional semantics and cross-lingual transfer.

This project addresses the following questions:

1. Do pretrained language models encode idiomatic meaning?
2. Does **idiom–meaning supervision** improve semantic retrieval?
3. How well do embeddings **transfer across Turkic languages**?
4. Can **few-shot supervision** close the zero-shot performance gap?

---

## 🧠 Tasks

### **Task 1 — Cross-Lingual Idiom Retrieval**
Retrieve the correct idiom in a target Turkic language given a Turkish idiom.

### **Task 2 — Monolingual Idiom → Meaning Retrieval**
Retrieve the correct meaning given an idiom in the same language.

### **Task 3 — Cross-Lingual Idiom → Meaning Retrieval**
Retrieve the meaning of an idiom across languages (e.g., TR idiom → AZ meaning).

### **Task 4 — Contextual Idiom Meaning Retrieval (Optional)**
Retrieve meanings using **sentential context** instead of isolated idioms.

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

The framework supports any HuggingFace embedding model.

Evaluated model families include:

- **Retrieval-oriented encoders**
  - Turkish-e5-large
  - Multilingual-e5
- **General multilingual encoders**
  - XLM-R
  - mBERT
  - MPNet
- **Turkish-specific encoders**
  - BERTurk
  - ConvBERT-tr
  - ELECTRA-tr
- **Sentence embedding baselines**
  - paraphrase-multilingual-mpnet-v2

Pooling strategies (configurable):
- CLS
- Mean pooling
- Sum pooling
- First–Last layer averaging
- Weighted sum pooling

---

## 🧪 Evaluation Metrics

Standard retrieval metrics:

- Recall@k
- MRR
- NDCG@k
- Hits@k

Metrics are reported:
- Per language
- Per task
- Per model
- Pretrained vs fine-tuned

