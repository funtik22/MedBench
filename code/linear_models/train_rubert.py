import json
import pathlib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

SEED = 128
torch.manual_seed(SEED)
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
    # Данные
    # -------------------------
    base_path = pathlib.Path(__file__).absolute().parent.parent.parent
    data_path = base_path / 'data' / 'RuMedTop3'

    train_data = pd.read_json(data_path / 'train_v1.jsonl', lines=True)
    dev_data = pd.read_json(data_path / 'dev_v1.jsonl', lines=True)
    test_data = pd.read_json(data_path / 'test_v1.jsonl', lines=True)

    texts_train = train_data['symptoms'].tolist()
    texts_dev = dev_data['symptoms'].tolist()
    texts_test = test_data['symptoms'].tolist()

    le = LabelEncoder()
    y_train = le.fit_transform(train_data['code'])
    y_dev = le.transform(dev_data['code'])
    num_labels = len(le.classes_)

    # -------------------------
    # Transformer
    # -------------------------
    model_name = 'ai-forever/rubert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_encodings = tokenizer(texts_train, truncation=True, padding=True, max_length=256)
    dev_encodings = tokenizer(texts_dev, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(texts_test, truncation=True, padding=True, max_length=256)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset = Dataset(train_encodings, y_train)
    dev_dataset = Dataset(dev_encodings, y_dev)
    test_dataset = Dataset(test_encodings)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    trainer.train()

    # -------------------------
    # Сохранение обученной модели и токенизатора
    # -------------------------
    out_path = base_path / 'code' / 'linear_models' / 'out'
    out_path.mkdir(parents=True, exist_ok=True)
    model_out_path = out_path / 'rubert_model'
    model_out_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_out_path)
    tokenizer.save_pretrained(model_out_path)

    print(f"RuBERT model and tokenizer saved to {model_out_path}")

    # -------------------------
    # Предсказания
    # -------------------------
    y_dev_logits = trainer.predict(dev_dataset).predictions
    y_dev_proba = torch.nn.functional.softmax(torch.tensor(y_dev_logits), dim=1).numpy()
    hit1 = hit_at_n(y_dev, y_dev_proba, n=1)
    hit3 = hit_at_n(y_dev, y_dev_proba, n=3)
    print(f"RuBERT Hit@1: {hit1}%, Hit@3: {hit3}%")

    y_test_logits = trainer.predict(test_dataset).predictions
    y_test_proba = torch.nn.functional.softmax(torch.tensor(y_test_logits), dim=1).numpy()

    test_codes = []
    for row in y_test_proba:
        top3_idx = np.argsort(row)[::-1][:3]
        test_codes.append([le.inverse_transform([i])[0] for i in top3_idx])


    out_fname = out_path / 'RuMedTop3_rubert.jsonl'
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
