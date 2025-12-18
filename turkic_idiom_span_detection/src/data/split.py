import pandas as pd
from sklearn.model_selection import train_test_split

def split_by_idiom_id(
    ner_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    if "idiom_id" not in ner_df.columns:
        raise ValueError("ner_df must include idiom_id column")

    unique_ids = ner_df["idiom_id"].unique().tolist()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=seed)

    def assign(iid: int) -> str:
        if iid in train_ids:
            return "train"
        if iid in val_ids:
            return "validation"
        return "test"

    ner_df = ner_df.copy()
    ner_df["split"] = ner_df["idiom_id"].apply(assign)
    return ner_df
