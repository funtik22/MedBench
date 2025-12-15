import json
import pathlib
from typing import List, Sequence, TYPE_CHECKING

import click
import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


SEED = 128
torch.manual_seed(SEED)
np.random.seed(SEED)


def hit_at_n(y_true: Sequence[int], y_pred_proba: np.ndarray, n: int = 3) -> float:
    """Hit@N metric for probability matrices."""
    top_n = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :n]
    hits = sum(int(label in row) for label, row in zip(y_true, top_n))
    return round(hits / len(y_true) * 100, 2)


def top_k_codes(probabilities: np.ndarray, label_encoder: LabelEncoder, k: int = 3) -> List[List[str]]:
    """Convert probability matrix to top-k label names."""
    predictions: List[List[str]] = []
    for row in probabilities:
        order = np.argsort(row)[::-1][:k]
        labels = label_encoder.inverse_transform(order)
        predictions.append(labels.tolist())
    return predictions


def blend_probabilities(probabilities: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted soft-voting for probability matrices."""
    if not probabilities:
        raise ValueError("No probability matrices passed to the ensemble")
    if len(probabilities) != len(weights):
        raise ValueError("Weights count must match number of probability matrices")

    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights = normalized_weights / normalized_weights.sum()

    blended = np.zeros_like(probabilities[0], dtype=float)
    for weight, proba in zip(normalized_weights, probabilities):
        blended += weight * proba
    return blended


def load_ru_med_top3(base_path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all splits for the RuMedTop3 task."""
    data_dir = base_path / "data" / "RuMedTop3"
    train = pd.read_json(data_dir / "train_v1.jsonl", lines=True)
    dev = pd.read_json(data_dir / "dev_v1.jsonl", lines=True)
    test = pd.read_json(data_dir / "test_v1.jsonl", lines=True)
    return train, dev, test


class HFDataset(torch.utils.data.Dataset):
    """Simple HF dataset wrapper to reuse tokenizer encodings."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


def predict_with_rubert(model, tokenizer: "AutoTokenizer", texts: Sequence[str], batch_size: int) -> np.ndarray:
    """Inference helper for a pre-trained RuBERT classifier."""
    # Lazy import to avoid TensorFlow/Keras dependency when RuBERT branch is skipped.
    from transformers import Trainer, TrainingArguments

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
    dataset = HFDataset(encodings)
    training_args = TrainingArguments(
        output_dir="ensemble_rubert_loaded",
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        do_eval=False,
        do_train=False,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args)
    logits = trainer.predict(dataset).predictions
    return torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()


def parse_weights(weights_raw: str, expected_len: int) -> List[float]:
    parts = [float(w.strip()) for w in weights_raw.split(",") if w.strip()]
    if not parts:
        raise ValueError("Weights string is empty")
    if len(parts) == 1:
        parts = parts * expected_len
    if len(parts) > expected_len:
        parts = parts[:expected_len]
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} weights, got {len(parts)}")
    return parts


def ensure_artifact(path: pathlib.Path, description: str, generation_hint: str) -> pathlib.Path:
    """Fail-fast helper that nudges to the right training script."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}. {generation_hint}")
    return path


def predict_lgbm_proba(model: lgb.Booster, X) -> np.ndarray:
    proba = np.asarray(model.predict(X))
    if proba.ndim == 1:
        raise ValueError("LightGBM returned 1D predictions, expected 2D probability matrix.")
    return proba


if TYPE_CHECKING:
    from transformers import AutoTokenizer  # noqa: F401


@click.command()
@click.option(
    "--rubert-dir",
    default="rubert_model",
    show_default=True,
    help="Directory inside code/linear_models/out with a saved RuBERT checkpoint.",
)
@click.option(
    "--rubert-batch-size",
    default=16,
    show_default=True,
    help="Batch size for RuBERT inference.",
)
@click.option(
    "--weights",
    default="1",
    show_default=True,
    help="Comma-separated weights for active models (logreg + catboost + lgbm [+rubert]).",
)
@click.option(
    "--skip-rubert",
    is_flag=True,
    default=False,
    help="Use only TF-IDF-based models (logreg, catboost, lgbm).",
)
def main(rubert_dir: str, rubert_batch_size: int, weights: str, skip_rubert: bool):
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    out_dir = project_root / "code" / "linear_models" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    _train_df, dev_df, test_df = load_ru_med_top3(project_root)

    text_col, label_col, idx_col = "symptoms", "code", "idx"
    artifacts_dir = out_dir
    tfidf_path = ensure_artifact(
        artifacts_dir / "tfidf_vectorizer.pkl",
        "TF-IDF vectorizer",
        "Generate via: python code/linear_models/log_reg.py",
    )
    label_encoder_path = ensure_artifact(
        artifacts_dir / "label_encoder.pkl",
        "LabelEncoder",
        "Generate via: python code/linear_models/log_reg.py",
    )
    logreg_path = ensure_artifact(
        artifacts_dir / "RuMedTop3_logreg.pkl",
        "Logistic regression model",
        "Generate via: python code/linear_models/log_reg.py",
    )
    catboost_path = ensure_artifact(
        artifacts_dir / "RuMedTop3_catboost.cbm",
        "CatBoost model",
        "Generate via: python code/linear_models/train_catboost.py",
    )
    lgbm_path = ensure_artifact(
        artifacts_dir / "RuMedTop3_lightgbm.txt",
        "LightGBM model",
        "Generate via: python code/linear_models/train_lightgbm.py",
    )

    tfidf = joblib.load(tfidf_path)
    label_encoder: LabelEncoder = joblib.load(label_encoder_path)

    X_dev = tfidf.transform(dev_df[text_col])
    X_test = tfidf.transform(test_df[text_col])
    y_dev = label_encoder.transform(dev_df[label_col])

    tfidf_dev_probabilities = []
    tfidf_test_probabilities = []
    model_names = []

    logreg_model = joblib.load(logreg_path)
    tfidf_dev_probabilities.append(logreg_model.predict_proba(X_dev))
    tfidf_test_probabilities.append(logreg_model.predict_proba(X_test))
    model_names.append("logreg")

    catboost_model = CatBoostClassifier()
    catboost_model.load_model(str(catboost_path))
    catboost_dev = np.asarray(catboost_model.predict_proba(X_dev))
    catboost_test = np.asarray(catboost_model.predict_proba(X_test))
    tfidf_dev_probabilities.append(catboost_dev)
    tfidf_test_probabilities.append(catboost_test)
    model_names.append("catboost")

    lgbm_model = lgb.Booster(model_file=str(lgbm_path))
    lgbm_dev = predict_lgbm_proba(lgbm_model, X_dev)
    lgbm_test = predict_lgbm_proba(lgbm_model, X_test)
    tfidf_dev_probabilities.append(lgbm_dev)
    tfidf_test_probabilities.append(lgbm_test)
    model_names.append("lgbm")

    for name, proba in zip(model_names, tfidf_dev_probabilities):
        hit1 = hit_at_n(y_dev, proba, n=1)
        hit3 = hit_at_n(y_dev, proba, n=3)
        print(f"{name} dev Hit@1/Hit@3: {hit1}/{hit3}")

    probabilities_dev = list(tfidf_dev_probabilities)
    probabilities_test = list(tfidf_test_probabilities)

    if not skip_rubert:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        rubert_path = ensure_artifact(
            artifacts_dir / rubert_dir,
            "RuBERT checkpoint",
            "Generate via: python code/linear_models/train_rubert.py",
        )
        rubert_model = AutoModelForSequenceClassification.from_pretrained(rubert_path)
        rubert_tokenizer = AutoTokenizer.from_pretrained(rubert_path)
        rubert_dev = predict_with_rubert(
            rubert_model,
            rubert_tokenizer,
            dev_df[text_col].tolist(),
            rubert_batch_size,
        )
        rubert_test = predict_with_rubert(
            rubert_model,
            rubert_tokenizer,
            test_df[text_col].tolist(),
            rubert_batch_size,
        )
        probabilities_dev.append(rubert_dev)
        probabilities_test.append(rubert_test)
        model_names.append("rubert")
        rubert_hit1 = hit_at_n(y_dev, rubert_dev, n=1)
        rubert_hit3 = hit_at_n(y_dev, rubert_dev, n=3)
        print(f"rubert dev Hit@1/Hit@3: {rubert_hit1}/{rubert_hit3}")
    else:
        print("Skipping RuBERT branch (expects pre-trained checkpoint under code/linear_models/out).")

    weights_to_use = parse_weights(weights, expected_len=len(probabilities_dev))
    ensemble_dev = blend_probabilities(probabilities_dev, weights_to_use)
    ensemble_test = blend_probabilities(probabilities_test, weights_to_use)

    ensemble_hit1 = hit_at_n(y_dev, ensemble_dev, n=1)
    ensemble_hit3 = hit_at_n(y_dev, ensemble_dev, n=3)
    print(f"Ensemble dev Hit@1/Hit@3: {ensemble_hit1}/{ensemble_hit3}")

    test_predictions = top_k_codes(ensemble_test, label_encoder, k=3)
    out_file = out_dir / "RuMedTop3_ensemble.jsonl"
    with open(out_file, "w", encoding="utf-8") as fw:
        for idx_value, true_code, pred in zip(test_df[idx_col], test_df[label_col], test_predictions):
            json.dump({"idx": idx_value, "code": true_code, "prediction": pred}, fw, ensure_ascii=False)
            fw.write("\n")
    print(f"Ensemble predictions saved to {out_file}")


if __name__ == "__main__":
    main()
