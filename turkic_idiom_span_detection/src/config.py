LANGS = ["TR", "AZ", "TK", "GA", "UZ"]

LABEL_LIST = ["O", "B-IDIOM", "I-IDIOM"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

RANDOM_SEED = 42
MAX_LENGTH = 256