from config import LABEL2ID

class TokenAligner:
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        tokenized = self.tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

        labels = []
        for i, labs in enumerate(examples["labels"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev = None
            label_ids = []

            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                else:
                    lab = labs[wid]
                    if wid == prev and lab.startswith("B-"):
                        lab = lab.replace("B-", "I-")
                    label_ids.append(LABEL2ID[lab])
                prev = wid

            tokenized["labels"] = label_ids
            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized
