import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SemanticEncoder:
    """
    Universal encoder:
    - SentenceTransformer → native
    - HF AutoModel → manual pooling
    """

    def __init__(self, model_id, pooling="mean", base_model_id=None):
        self.model_id = model_id
        self.pooling = pooling
        self.base_model_id = base_model_id or model_id

        # Try ST
        try:
            self.st = SentenceTransformer(model_id, device=DEVICE)
            self.backend = "st"
            return
        except Exception:
            self.backend = "hf"

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            output_hidden_states=True
        ).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        if self.backend == "st":
            return self.st.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True
            )

        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(DEVICE)

            out = self.model(**tok).last_hidden_state
            pooled = out.mean(dim=1)
            pooled = pooled / pooled.norm(dim=1, keepdim=True)
            embs.append(pooled.cpu().numpy())

        return np.vstack(embs)
