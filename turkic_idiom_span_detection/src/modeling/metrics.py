import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from config import ID2LABEL

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)

    true_preds, true_labels = [], []
    for p_row, l_row in zip(preds, labels):
        p_seq, l_seq = [], []
        for p_i, l_i in zip(p_row, l_row):
            if l_i != -100:
                p_seq.append(ID2LABEL[p_i])
                l_seq.append(ID2LABEL[l_i])
        true_preds.append(p_seq)
        true_labels.append(l_seq)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }
