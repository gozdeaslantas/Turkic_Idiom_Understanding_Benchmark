from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from modeling.metrics import compute_metrics
from modeling.callbacks import EpochLogger
from utils.seed import set_seed
from config import LABEL_LIST, LABEL2ID, ID2LABEL

class TurkishTrainer:
    def __init__(self, model_name, out_dir):
        self.model_name = model_name
        self.out_dir = out_dir

    def train(self, train_ds, val_ds):
        set_seed()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABEL_LIST),
            label2id=LABEL2ID,
            id2label=ID2LABEL
        )

        args = TrainingArguments(
            output_dir=self.out_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=5,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EpochLogger(f"{self.out_dir}/log.json")]
        )

        trainer.train()
        return trainer
