"""Inference wrapper around the fine-tuned sentiment model."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import DEFAULT_MODEL_DIR


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentimentClassifier:
    """Loads a fine-tuned model once and runs batched predictions."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, batch_size: int = 32):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No model found at {model_dir}. Train one first with "
                "`python -m src.train`."
            )
        self.device = _resolve_device()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[dict]:
        """Return label + per-class probabilities for each input text."""
        results: List[dict] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=-1).cpu()
            for row in probs:
                top = int(torch.argmax(row))
                results.append(
                    {
                        "label": self.id2label[top],
                        "score": float(row[top]),
                        "probabilities": {
                            self.id2label[i]: float(p) for i, p in enumerate(row)
                        },
                    }
                )
        return results


@lru_cache(maxsize=1)
def get_classifier() -> SentimentClassifier:
    """Process-wide singleton so the model is loaded only once."""
    return SentimentClassifier()
