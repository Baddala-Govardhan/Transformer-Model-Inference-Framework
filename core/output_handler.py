"""Stage 4: Convert raw model output into a consistent, human-readable result."""

import torch


class OutputHandler:
    def __init__(self, processor, task: str):
        self.processor = processor
        self.task = task

    def run(self, raw_output, **kwargs) -> dict:
        if self.task in ("text-generation", "text-seq2seq"):
            decoded = self.processor.batch_decode(raw_output, skip_special_tokens=True)
            return {"task": self.task, "output": decoded}

        if self.task in ("text-classification", "vision", "audio"):
            logits = raw_output.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = int(torch.argmax(probs, dim=-1)[0])
            label = self.model_label(pred_id)
            return {
                "task": self.task,
                "predicted_label": label,
                "confidence": float(probs[0][pred_id]),
            }

        if self.task == "multimodal":
            return self._handle_multimodal(raw_output, kwargs.get("candidates", []))

        return {"task": self.task, "output_type": type(raw_output).__name__}

    def _handle_multimodal(self, raw_output, candidates: list[str]) -> dict:
        # CLIP-style output: logits_per_image gives an image-to-text similarity
        # score per candidate text; softmax turns those into comparable probabilities.
        logits_per_image = raw_output.logits_per_image
        probs = torch.softmax(logits_per_image, dim=-1)[0]
        scores = {label: float(p) for label, p in zip(candidates, probs.tolist())}
        best_match = max(scores, key=scores.get) if scores else None
        return {"task": self.task, "best_match": best_match, "scores": scores}

    def model_label(self, pred_id: int) -> str:
        id2label = getattr(self.processor, "id2label", None)
        return id2label[pred_id] if id2label else str(pred_id)
