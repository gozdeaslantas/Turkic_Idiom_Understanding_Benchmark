def build_task1(df, src="TR", tgts=("AZ","TK","GA","UZ")):
    pairs = []
    for gid in df["group_id"].unique():
        src_row = df[(df.group_id==gid) & (df.lang==src)]
        if src_row.empty:
            continue
        src_idiom = src_row.iloc[0]["idiom"]

        for L in tgts:
            tgt = df[(df.group_id==gid) & (df.lang==L)]
            if tgt.empty:
                continue
            pairs.append({
                "group_id": gid,
                "src_lang": src,
                "tgt_lang": L,
                "src_text": src_idiom,
                "tgt_text": tgt.iloc[0]["idiom"]
            })
    return pairs


def build_task2(df):
    return [
        {
            "group_id": r.group_id,
            "lang": r.lang,
            "src_text": r.idiom,
            "tgt_text": r.meaning
        }
        for _, r in df.iterrows()
        if isinstance(r.idiom, str) and isinstance(r.meaning, str)
    ]


def build_task3(df, src="TR", tgts=("AZ", "TK", "GA", "UZ")):
    pairs = []

    for gid in df["group_id"].unique():
        src_row = df[(df.group_id == gid) & (df.lang == src)]
        if src_row.empty:
            continue

        src_idiom = src_row.iloc[0]["idiom"]
        if not isinstance(src_idiom, str):
            continue

        for L in tgts:
            tgt_row = df[(df.group_id == gid) & (df.lang == L)]
            if tgt_row.empty:
                continue

            tgt_meaning = tgt_row.iloc[0]["meaning"]
            if not isinstance(tgt_meaning, str):
                continue

            pairs.append({
                "group_id": gid,
                "src_lang": src,
                "tgt_lang": L,
                "src_text": src_idiom,     
                "tgt_text": tgt_meaning    
            })

    return pairs
)

