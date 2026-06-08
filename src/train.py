"""Fine-tune DistilBERT on the Financial PhraseBank and track runs with MLflow.

Usage:
    python -m src.train --epochs 4 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import ID2LABEL, LABEL2ID, PROJECT_ROOT, TrainConfig
from .data import label_names, load_splits

RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for financial sentiment.")
    parser.add_argument("--base-model", default=cfg.base_model)
    parser.add_argument("--epochs", type=int, default=cfg.num_train_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.train_batch_size)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    cfg.base_model = args.base_model
    cfg.num_train_epochs = args.epochs
    cfg.train_batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.max_length = args.max_length
    cfg.seed = args.seed
    return cfg


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_compute_metrics(labels: list[str]):
    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(gold, preds),
            "f1_macro": f1_score(gold, preds, average="macro"),
            "f1_weighted": f1_score(gold, preds, average="weighted"),
        }

    return compute_metrics


def save_confusion_matrix(gold, preds, labels: list[str], path: Path) -> None:
    cm = confusion_matrix(gold, preds, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Using device: {device}")

    splits = load_splits(cfg)
    labels = label_names(splits)
    print({split: len(ds) for split, ds in splits.items()})

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=cfg.max_length)

    tokenized = splits.map(tokenize, batched=True, remove_columns=["sentence"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(labels),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir / "checkpoints"),
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        seed=cfg.seed,
        report_to=[],  # MLflow logging handled explicitly below.
        use_cpu=device == "cpu",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(labels),
    )

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "base_model": cfg.base_model,
                "dataset_source": cfg.dataset_source,
                "epochs": cfg.num_train_epochs,
                "learning_rate": cfg.learning_rate,
                "train_batch_size": cfg.train_batch_size,
                "max_length": cfg.max_length,
                "seed": cfg.seed,
                "device": device,
                "n_train": len(tokenized["train"]),
                "n_val": len(tokenized["validation"]),
                "n_test": len(tokenized["test"]),
            }
        )

        trainer.train()

        # Final evaluation on the held-out test set.
        predictions = trainer.predict(tokenized["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        gold = predictions.label_ids

        test_metrics = {
            "test_accuracy": accuracy_score(gold, preds),
            "test_f1_macro": f1_score(gold, preds, average="macro"),
            "test_f1_weighted": f1_score(gold, preds, average="weighted"),
        }
        mlflow.log_metrics(test_metrics)

        report = classification_report(gold, preds, target_names=labels, digits=4)
        print(report)
        report_dict = classification_report(
            gold, preds, target_names=labels, output_dict=True
        )

        cm_path = RESULTS_DIR / "confusion_matrix.png"
        save_confusion_matrix(gold, preds, labels, cm_path)
        mlflow.log_artifact(str(cm_path))

        results_path = RESULTS_DIR / "metrics.json"
        results_path.write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "params": {
                        "base_model": cfg.base_model,
                        "dataset_source": cfg.dataset_source,
                        "epochs": cfg.num_train_epochs,
                    },
                    "test_metrics": test_metrics,
                    "classification_report": report_dict,
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(results_path))

        # Persist the fine-tuned model + tokenizer for serving.
        trainer.save_model(str(cfg.output_dir))
        tokenizer.save_pretrained(str(cfg.output_dir))
        mlflow.log_artifacts(str(cfg.output_dir), artifact_path="model")

        print(f"\nTest metrics: {json.dumps(test_metrics, indent=2)}")
        print(f"Model saved to: {cfg.output_dir}")
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
