import json
import pathlib
import click
import numpy as np
import pandas as pd
import time
import joblib

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def hit_at_n(y_true, y_pred, n=3):
    hit = 0
    for y, row in zip(y_true, y_pred):
        topn = np.argsort(row)[::-1][:n]
        hit += int(y in topn)
    return round(hit / len(y_true) * 100, 2)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return softmax(model.decision_function(X))


def logits2codes(logits, i2l, n=3):
    preds = []
    for row in logits:
        order = np.argsort(row)[::-1][:n]
        preds.append([i2l[i] for i in order])
    return preds


def rank_ensemble(models, weights, X, num_classes):
    scores = np.zeros((X.shape[0], num_classes))

    for w, model in zip(weights, models):
        probs = get_proba(model, X)
        ranks = np.argsort(-probs, axis=1)

        for i in range(X.shape[0]):
            for r, cls in enumerate(ranks[i]):
                scores[i, cls] += w / (r + 1)

    return scores


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            C=10,
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                max_iter=1000,
                class_weight="balanced"
            ),
            cv=3
        ),
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
    }


def train_and_eval(name, model, X_train, y_train, X_val, y_val):
    print(f"\nОбучение: {name}")
    start = time.time()

    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 2)

    val_pred = get_proba(model, X_val)
    hit3 = hit_at_n(y_val, val_pred, n=3)

    print(f"  Hit@3: {hit3}%")
    print(f"  Время: {train_time}s")

    return {
        "name": name,
        "model": model,
        "hit@3": hit3
    }


@click.command()
@click.option("--task-name", default="RuMedTop3", type=click.Choice(["RuMedTop3"]))
def main(task_name):

    print("=" * 80)
    print(f"{task_name}")
    print("=" * 80)

    base = pathlib.Path(__file__).absolute().parent.parent.parent
    data_path = base / "data" / task_name
    out_path = base / "code" / "linear_models" / "out"
    out_path.mkdir(parents=True, exist_ok=True)

    train = pd.read_json(data_path / "train_v1.jsonl", lines=True)
    dev = pd.read_json(data_path / "dev_v1.jsonl", lines=True)
    test = pd.read_json(data_path / "test_v1.jsonl", lines=True)

    print(f'Train size: {len(train)}') 
    print(f'Dev size: {len(dev)}') 
    print(f'Test size: {len(test)}')

    text_id = "symptoms"
    label_id = "code"
    idx_id = "idx"

    i2l = dict(enumerate(sorted(train[label_id].unique())))
    l2i = {v: k for k, v in i2l.items()}
    print(f'Количество классов: {len(i2l)}')

    y_train = train[label_id].map(l2i)
    y_val = dev[label_id].map(l2i)

    tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 8), max_features=50000)
    tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=30000)

    Xc_train = tfidf_char.fit_transform(train[text_id])
    Xw_train = tfidf_word.fit_transform(train[text_id])
    X_train = hstack([Xc_train, Xw_train])

    Xc_val = tfidf_char.transform(dev[text_id])
    Xw_val = tfidf_word.transform(dev[text_id])
    X_val = hstack([Xc_val, Xw_val])

    Xc_test = tfidf_char.transform(test[text_id])
    Xw_test = tfidf_word.transform(test[text_id])
    X_test = hstack([Xc_test, Xw_test])
    
    print(f'Размер матрицы признаков: {X_train.shape}')
    
    models = get_models()
    results = []

    for name, model in models.items():
        res = train_and_eval(name, model, X_train, y_train, X_val, y_val)
        results.append(res)

    weights = np.array([r["hit@3"] for r in results])
    for i, r in enumerate(results):
        if r["name"] == "RandomForest":
            weights[i] *= 0.7
    weights = weights / weights.sum()

    ensemble_val = rank_ensemble(
        [r["model"] for r in results],
        weights,
        X_val,
        len(i2l)
    )

    ens_acc1 = hit_at_n(y_val, ensemble_val, n=1)
    ens_hit3 = hit_at_n(y_val, ensemble_val, n=3)

    print(f"\n\nensemble Accuracy@1: {ens_acc1}%")
    print(f"ensemble Hit@3: {ens_hit3}%")

    ensemble_test = rank_ensemble(
        [r["model"] for r in results],
        weights,
        X_test,
        len(i2l)
    )

    preds = logits2codes(ensemble_test, i2l)

    out_file = out_path / "RuMedTop3.jsonl"
    with open(out_file, "w", encoding="utf8") as f:
        for idx, true, pred in zip(test[idx_id], test[label_id], preds):
            json.dump(
                {
                    idx_id: idx,
                    label_id: true,
                    "prediction": pred
                },
                f,
                ensure_ascii=False
            )
            f.write("\n")

    print(f"\nПредсказания сохранены в: {out_file}")

    joblib.dump(tfidf_char, out_path / "tfidf_char.pkl")
    joblib.dump(tfidf_word, out_path / "tfidf_word.pkl")
    joblib.dump(results, out_path / "models.pkl")
    joblib.dump(i2l, out_path / "i2l.pkl")
    joblib.dump(weights, out_path / "ensemble_weights.pkl")

    print("\nАртефакты сохранены:")
    print("  tfidf_char.pkl")
    print("  tfidf_word.pkl")
    print("  models.pkl")
    print("  i2l.pkl")
    print("  ensemble_weights.pkl")


if __name__ == "__main__":
    main()
