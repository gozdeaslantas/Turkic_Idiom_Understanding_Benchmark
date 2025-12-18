import pandas as pd
from tqdm import tqdm
from .metrics import compute_single_query_metrics

def evaluate_task(encoder, df_long, pairs, candidate_col):
    results = []
    cache = {}

    for p in tqdm(pairs):
        L = p.get("tgt_lang", p.get("lang"))
        if L not in cache:
            pool = df_long[(df_long.lang==L) & df_long[candidate_col].notna()]
            texts = pool[candidate_col].tolist()
            embs = encoder.encode(texts)
            cache[L] = (texts, embs)

        texts, embs = cache[L]
        if p["tgt_text"] not in texts:
            continue

        q = encoder.encode([p["src_text"]])[0]
        idx = texts.index(p["tgt_text"])
        m = compute_single_query_metrics(q, embs, idx)
        results.append(m)

    return pd.DataFrame(results)
