import pandas as pd
import numpy as np

LANGS = ["TR", "AZ", "TK", "GA", "UZ"]

def build_long_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []

    for idx, r in df.iterrows():
        for L in LANGS:
            idiom = r.get(f"{L}_Deyim")
            meaning = r.get(f"{L}_Anlam")
            example = r.get(f"{L}_Ornek")

            if pd.isna(idiom) and pd.isna(meaning):
                continue

            rows.append({
                "group_id": idx,
                "lang": L,
                "idiom": None if pd.isna(idiom) else str(idiom).strip(),
                "meaning": None if pd.isna(meaning) else str(meaning).strip(),
                "example": "" if pd.isna(example) else str(example)
            })

    return pd.DataFrame(rows)
