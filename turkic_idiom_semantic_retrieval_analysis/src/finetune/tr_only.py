from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def run_tr_only_ft(base_model, df_train, out_dir, epochs=1):
    pairs = [
        (r.idiom, r.meaning)
        for _, r in df_train[df_train.lang=="TR"].iterrows()
        if isinstance(r.idiom,str) and isinstance(r.meaning,str)
    ]

    model = SentenceTransformer(base_model)
    examples = [InputExample(texts=[a,b]) for a,b in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        show_progress_bar=True
    )
    model.save(out_dir)
    return out_dir
