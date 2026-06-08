"""Central configuration for the financial sentiment classifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_DIR = MODELS_DIR / "distilbert-financial-sentiment"

# Label space for the Financial PhraseBank dataset.
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


@dataclass
class TrainConfig:
    """Hyperparameters and data settings for fine-tuning."""

    base_model: str = "distilbert-base-uncased"
    # Authentic Financial PhraseBank mirror with standard train/val/test splits.
    dataset_source: str = "ChanceFocus/flare-fpb"
    max_length: int = 128
    seed: int = 42

    num_train_epochs: int = 4
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    output_dir: Path = field(default_factory=lambda: DEFAULT_MODEL_DIR)
    mlflow_experiment: str = "financial-sentiment"
    # MLflow 3.x deprecated the plain file store; use the recommended SQLite
    # backend. Artifacts land under ./mlartifacts.
    mlflow_tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
