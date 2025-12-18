import pandas as pd
from data.ner_labeler import IdiomSpanLabeler
from config import LANGS

class DatasetBuilder:
    def __init__(self):
        self.labeler = IdiomSpanLabeler()

    def build(self, df: pd.DataFrame):
        records = []

        for idx, row in df.iterrows():
            for L in LANGS:
                sent = row.get(f"{L}_Ornek")
                idiom = row.get(f"{L}_Deyim")
                if pd.isna(sent) or pd.isna(idiom):
                    continue

                rec = self.labeler.label(sent, idiom, L, idx)
                if rec:
                    records.append(rec)

        return pd.DataFrame(records)
