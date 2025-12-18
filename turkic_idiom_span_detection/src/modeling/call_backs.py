import os
import json
from transformers import TrainerCallback

class EpochLogger(TrainerCallback):
    def __init__(self, out_path):
        self.out_path = out_path
        self.history = []

    def _save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.history.append({
                "epoch": state.epoch,
                "step": state.global_step,
                "logs": logs
            })
            with open(self.out_path, "w") as f:
                json.dump(self.history, f, indent=2)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        item = {
            "event": "eval",
            "epoch": state.epoch,
            "step": state.global_step,
            "metrics": metrics
        }
        self.history.append(item)
        self._save()