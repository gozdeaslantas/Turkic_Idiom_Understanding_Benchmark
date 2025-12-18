def is_sentence_transformer(model_id: str) -> bool:
    return (
        "sentence-transformers" in model_id
        or "e5" in model_id
        or "mpnet" in model_id
    )
