"""Stage 3: Run the forward pass or generation against the loaded model."""

import torch


class InferenceEngine:
    def __init__(self, model, task: str):
        self.model = model
        self.task = task

    @torch.no_grad()
    def run(self, model_inputs: dict, **generate_kwargs):
        if self.task in ("text-generation", "text-seq2seq"):
            return self.model.generate(**model_inputs, **generate_kwargs)
        # classification / embedding / multimodal tasks use a plain forward pass
        return self.model(**model_inputs)
