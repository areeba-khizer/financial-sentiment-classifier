"""Load and prepare the Financial PhraseBank dataset.

The Financial PhraseBank (Malo et al., 2014) contains ~4.8k financial news
sentences hand-labelled by domain experts as positive / negative / neutral. We
source it from the `ChanceFocus/flare-fpb` mirror, which ships authentic
sentences as parquet files with standard train / validation / test splits.
"""
from __future__ import annotations

from datasets import DatasetDict, load_dataset

from .config import LABEL2ID, TrainConfig

_SOURCE = "ChanceFocus/flare-fpb"


def load_splits(cfg: TrainConfig) -> DatasetDict:
    """Return a DatasetDict with `sentence` and integer `label` columns."""
    raw = load_dataset(_SOURCE)

    def to_example(row):
        # `answer` holds the canonical label string (positive/negative/neutral).
        return {"sentence": row["text"], "label": LABEL2ID[row["answer"].strip().lower()]}

    prepared = DatasetDict()
    rename = {"train": "train", "valid": "validation", "test": "test"}
    for src_split, dst_split in rename.items():
        prepared[dst_split] = raw[src_split].map(
            to_example, remove_columns=raw[src_split].column_names
        )
    return prepared


def label_names(splits: DatasetDict) -> list[str]:  # noqa: ARG001 - kept for API symmetry
    from .config import ID2LABEL

    return [ID2LABEL[i] for i in sorted(ID2LABEL)]
