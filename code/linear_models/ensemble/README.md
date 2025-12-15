# Руководство по запуску ансамбля RuMedTop3

`top3_ensemble.py` берет уже обученные модели из `code/linear_models/out`, усредняет их вероятности (soft voting) и сохраняет топ‑3 предсказания.

## Что нужно подготовить
- TF-IDF + `RuMedTop3_logreg.pkl` + `label_encoder.pkl`: `python code/linear_models/log_reg.py`
- `RuMedTop3_catboost.cbm`: `python code/linear_models/train_catboost.py`
- `RuMedTop3_lightgbm.txt`: `python code/linear_models/train_lightgbm.py`
- (опционально) директория `rubert_model`: `python code/linear_models/train_rubert.py`

Все файлы должны лежать в `code/linear_models/out` перед запуском ансамбля.

## Используемые модели и параметры
- `logreg`: `LogisticRegression` (`C=10`, `multi_class="ovr"`, `max_iter=1000`, `solver="lbfgs"`, `n_jobs=10`) на символьном TF-IDF.
- `catboost`: `CatBoostClassifier` (`iterations=400`, `learning_rate=0.1`, `depth=6`, `loss_function="MultiClass"`, `random_seed=128`, `verbose=50`).
- `lgbm`: `LGBMClassifier` (`objective="multiclass"`, `num_class=<кол-во классов>`, `n_estimators=400`, `learning_rate=0.1`, `random_state=128`).
- `rubert` (опционально): загружается из сохраненной директории, инференс батч задается `--rubert-batch-size`.

## Запуск ансамбля
Из корня репозитория после подготовки артефактов:
- Только TF-IDF модели:  
  `python code/linear_models/ensemble/top3_ensemble.py --skip-rubert --weights 1,1,1`
- С RuBERT:  
  `python code/linear_models/ensemble/top3_ensemble.py --weights 1,1,1,1`

## Опции
- `--weights`: веса в порядке `logreg,catboost,lgbm[,rubert]`. Один вес дублируется для всех.
- `--skip-rubert`: не подключать сохраненный RuBERT чекпойнт.
- `--rubert-dir`: название директории с чекпойнтом RuBERT внутри `code/linear_models/out` (по умолчанию `rubert_model`).
- `--rubert-batch-size`: батч на инференсе RuBERT.

## Выходные данные
- Метрики Hit@1/Hit@3 печатаются по каждой базовой модели и для итогового ансамбля.
- Предсказания сохраняются в `code/linear_models/out/RuMedTop3_ensemble.jsonl` с полями `idx`, `code`, `prediction` (список из 3 кодов).
