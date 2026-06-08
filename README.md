# Financial News Sentiment Classifier

[![CI](https://github.com/areeba-khizer/financial-sentiment-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/areeba-khizer/financial-sentiment-classifier/actions/workflows/ci.yml)

Fine-tuned **DistilBERT** that classifies financial news sentences as
**positive**, **negative**, or **neutral**. The project covers the full
applied-NLP workflow: data preparation, fine-tuning a transformer, tracking
experiments with **MLflow**, and serving predictions through a **FastAPI**
REST API with batch support.

## Highlights

- 🤗 **Transformers fine-tuning** — DistilBERT fine-tuned on the
  [Financial PhraseBank](https://huggingface.co/datasets/ChanceFocus/flare-fpb)
  (Malo et al., 2014), ~4.8k expert-labelled financial sentences.
- 📊 **MLflow experiment tracking** — every run logs hyperparameters, metrics,
  a confusion matrix, and the trained model as artifacts.
- ⚡ **FastAPI serving** — `/predict` endpoint accepts a batch of texts and
  returns labels with per-class probabilities.
- ✅ **Tested** — API contract covered by `pytest` using a stubbed model, so
  tests run without GPU or trained weights.

## Project layout

```
.
├── app/
│   └── main.py          # FastAPI service (health, labels, batch predict)
├── src/
│   ├── config.py        # hyperparameters, paths, label space
│   ├── data.py          # Financial PhraseBank loading + label mapping
│   ├── train.py         # fine-tuning loop + MLflow logging + evaluation
│   └── predict.py       # batched inference wrapper
├── tests/
│   └── test_api.py      # API tests with a stubbed classifier
├── results/             # metrics.json + confusion_matrix.png (committed)
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --epochs 4 --batch-size 16
```

This downloads the dataset, fine-tunes DistilBERT, evaluates on the held-out
test split, and writes:

- the model + tokenizer to `models/distilbert-financial-sentiment/`
- metrics and a classification report to `results/metrics.json`
- a confusion matrix to `results/confusion_matrix.png`
- a full MLflow run (params, metrics, artifacts) to `mlflow.db` / `mlartifacts/`

Training runs on GPU/Apple-Silicon MPS automatically when available, else CPU.

### Inspect experiments in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

## Serve

```bash
uvicorn app.main:app --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

### Example request

```bash
curl -s http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": [
        "The company beat earnings expectations and raised full-year guidance.",
        "Shares plunged after the firm cut its dividend.",
        "The board will meet next Tuesday to review the report."
      ]}' | python -m json.tool
```

```json
{
  "predictions": [
    {"text": "The company beat earnings expectations and raised full-year guidance.",
     "label": "positive", "score": 0.99, "probabilities": {"negative": 0.0, "neutral": 0.01, "positive": 0.99}},
    {"text": "Shares plunged after the firm cut its dividend.",
     "label": "negative", "score": 0.98, "probabilities": {"negative": 0.98, "neutral": 0.01, "positive": 0.01}},
    {"text": "The board will meet next Tuesday to review the report.",
     "label": "neutral", "score": 0.97, "probabilities": {"negative": 0.01, "neutral": 0.97, "positive": 0.02}}
  ]
}
```

## Results

DistilBERT fine-tuned for 4 epochs, evaluated on the held-out test split (970
sentences):

| Metric | Score |
| --- | --- |
| Accuracy | 0.858 |
| Macro F1 | 0.852 |
| Weighted F1 | 0.858 |

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.852 | 0.897 | 0.874 | 116 |
| neutral | 0.892 | 0.875 | 0.884 | 577 |
| positive | 0.791 | 0.805 | 0.798 | 277 |

Full metrics are tracked in [`results/metrics.json`](results/metrics.json) and
the confusion matrix in [`results/confusion_matrix.png`](results/confusion_matrix.png).

> Note: trained model weights are not committed (they're large and
> reproducible). Run `python -m src.train` to regenerate them locally.

## Test

```bash
pytest -q
```

Tests also run automatically on every push and pull request via
[GitHub Actions](.github/workflows/ci.yml).

## Run with Docker

The image bundles the API and its dependencies; mount your trained model at
runtime (weights aren't baked in, since they're large and reproducible):

```bash
docker build -t financial-sentiment .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" financial-sentiment
```

Then hit <http://127.0.0.1:8000/docs>.

## Dataset & license

Financial PhraseBank: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., &
Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in
economic texts.* Journal of the Association for Information Science and
Technology. Distributed for research use.
