import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

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
    # Загрузка данных
    # -------------------------
    base_path = pathlib.Path(__file__).absolute().parent.parent.parent
    data_path = base_path / 'data' / 'RuMedTop3'

    train_data = pd.read_json(data_path / 'train_v1.jsonl', lines=True)
    dev_data = pd.read_json(data_path / 'dev_v1.jsonl', lines=True)
    test_data = pd.read_json(data_path / 'test_v1.jsonl', lines=True)

    texts_train = train_data['symptoms'].tolist()
    texts_dev = dev_data['symptoms'].tolist()

    le = LabelEncoder()
    y_train = le.fit_transform(train_data['code'])
    y_dev = le.transform(dev_data['code'])

    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3,8))
    X_train = tfidf.fit_transform(texts_train)
    X_dev = tfidf.transform(texts_dev)

    # -------------------------
    # Обучение LightGBM
    # -------------------------
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        n_estimators=500,
        learning_rate=0.1,
        random_state=SEED
    )
    model.fit(X_train, y_train)

    # -------------------------
    # Сохранение модели LightGBM
    # -------------------------
    out_path = base_path / 'code' / 'linear_models' / 'out'
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / 'RuMedTop3_lightgbm.txt'
    model.booster_.save_model(str(model_path))
    print(f"LightGBM model saved to {model_path}")

    # -------------------------
    # Оценка на dev
    # -------------------------
    y_dev_pred_proba = model.predict_proba(X_dev)
    hit1 = hit_at_n(y_dev, y_dev_pred_proba, n=1)
    hit3 = hit_at_n(y_dev, y_dev_pred_proba, n=3)
    print(f"LightGBM Hit@1: {hit1}%, Hit@3: {hit3}%")

    # -------------------------
    # Предсказания на тесте
    # -------------------------
    X_test = tfidf.transform(test_data['symptoms'].tolist())
    y_test_proba = model.predict_proba(X_test)

    test_codes = []
    for row in y_test_proba:
        top3_idx = np.argsort(row)[::-1][:3]
        test_codes.append([le.inverse_transform([i])[0] for i in top3_idx])

    out_fname = out_path / 'RuMedTop3_lightgbm.jsonl'
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
