import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib  # Для сохранения модели

SEED = 128
np.random.seed(SEED)

# -------------------------
# Метрика Hit@3
# -------------------------
def hit_at_n(y_true, y_pred_proba, n=3):
    hits = 0
    for i, row in enumerate(y_pred_proba):
        top_n = np.argsort(row)[::-1][:n]
        if y_true[i] in top_n:
            hits += 1
    return round(hits / len(y_true) * 100, 2)

# -------------------------
# Основная функция
# -------------------------
def main():
    # -------------------------
    # Пути
    # -------------------------
    base_path = pathlib.Path(__file__).absolute().parent.parent.parent
    data_path = base_path / 'data' / 'RuMedTop3'
    out_path = base_path / 'code' / 'linear_models' / 'out'
    out_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Загрузка данных
    # -------------------------
    train_data = pd.read_json(data_path / 'train_v1.jsonl', lines=True)
    dev_data = pd.read_json(data_path / 'dev_v1.jsonl', lines=True)
    test_data = pd.read_json(data_path / 'test_v1.jsonl', lines=True)

    texts_train = train_data['symptoms'].tolist()
    texts_dev = dev_data['symptoms'].tolist()

    le = LabelEncoder()
    y_train = le.fit_transform(train_data['code'])
    y_dev = le.transform(dev_data['code'])

    # -------------------------
    # TF-IDF
    # -------------------------
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 8))
    X_train = tfidf.fit_transform(texts_train)
    X_dev = tfidf.transform(texts_dev)

    # -------------------------
    # Logistic Regression
    # -------------------------
    model = LogisticRegression(
        solver='lbfgs',  # solver lbfgs поддерживает мультиклассы
        max_iter=1000,
        random_state=SEED
    )
    model.fit(X_train, y_train)

    # -------------------------
    # Сохранение модели
    # -------------------------
    model_file = out_path / 'RuMedTop3_logreg.pkl'
    tfidf_file = out_path / 'tfidf_vectorizer.pkl'
    joblib.dump(model, model_file)
    joblib.dump(tfidf, tfidf_file)
    joblib.dump(le, out_path / 'label_encoder.pkl')
    print(f"Model saved to {model_file}")
    print(f"TF-IDF saved to {tfidf_file}")

    # -------------------------
    # Оценка на dev
    # -------------------------
    y_dev_proba = model.predict_proba(X_dev)
    hit1 = hit_at_n(y_dev, y_dev_proba, n=1)
    hit3 = hit_at_n(y_dev, y_dev_proba, n=3)
    print(f"Logistic Regression Hit@1: {hit1}%, Hit@3: {hit3}%")

    # -------------------------
    # Предсказания на тесте
    # -------------------------
    X_test = tfidf.transform(test_data['symptoms'].tolist())
    y_test_proba = model.predict_proba(X_test)

    test_codes = []
    for row in y_test_proba:
        top3_idx = np.argsort(row)[::-1][:3]
        test_codes.append([le.inverse_transform([i])[0] for i in top3_idx])

    out_fname = out_path / 'RuMedTop3_logreg.jsonl'
    with open(out_fname, 'w', encoding='utf-8') as fw:
        for idx, code, pred in zip(test_data['idx'], test_data['code'], test_codes):
            json.dump({'idx': idx, 'code': code, 'prediction': pred}, fw, ensure_ascii=False)
            fw.write('\n')

    print(f"Predictions saved to {out_fname}")

# -------------------------
# Точка входа
# -------------------------
if __name__ == "__main__":
    main()
