# Cold-Item Baseline

Это текущий baseline-модуль для сценария `cold-item`.

Он оставлен основной веткой после сравнения с retrieval + ranking экспериментом.

## Идея

Используется простой гибрид:

- `ALS` как collaborative модель
- `CatBoostRegressor` как feature-based модель

Во время обучения:

1. строится матрица `user-item`
2. items делятся на `warm` и `cold`
3. `ALS` обучается на interactions
4. `CatBoostRegressor` учится предсказывать `ALS score`

Во время инференса на паре `(user, item)` считаются:

- `als_score`
- `regressor_score`

Итоговый score:

```text
final_score = max(als_score, regressor_score)
```

## Что реально реализовано

По текущему коду baseline работает через внешний candidate pool.

Основной inference input:

```text
candidate_pairs.csv
```

с обязательными колонками:

- `user_id`
- `item_id`

То есть этот baseline не генерирует кандидатов сам, а перескоривает уже переданные пары.

## Структура

```text
cold-item/
├── README.md
├── main_train.py
├── main_infer.py
└── src/
    ├── als_model.py
    ├── data_loader.py
    ├── feature_builder.py
    ├── hybrid_recommender.py
    ├── inference_pipeline.py
    ├── preprocessing.py
    ├── ranker_model.py
    ├── split_warm_cold.py
    └── train_pipeline.py
```

## Что делает каждый блок

- `data_loader.py`
  - загружает CSV и проверяет обязательные колонки

- `preprocessing.py`
  - нормализует ids, выделяет `user_*` и `item_*` признаки, строит pair-features

- `split_warm_cold.py`
  - делит items на `warm/cold` по popularity

- `als_model.py`
  - обучает ALS и считает `ALS score`

- `feature_builder.py`
  - собирает train pairs и negative samples для регрессора

- `ranker_model.py`
  - содержит `CatBoostRegressor`

- `hybrid_recommender.py`
  - объединяет `ALS` и регрессор в один scorer

- `train_pipeline.py`
  - orchestration обучения

- `inference_pipeline.py`
  - orchestration inference по candidate pairs CSV

## Формат train CSV

Обязательные колонки:

- `user_id`
- `item_id`
- `value`

Опционально:

- `user_*`
- `item_*`

## CLI

Обучение:

```bash
python cold-item/main_train.py \
  --train-csv path/to/train.csv \
  --model-output artifacts/hybrid_model.joblib
```

Инференс:

```bash
python cold-item/main_infer.py \
  --model-path artifacts/hybrid_model.joblib \
  --input-csv path/to/candidate_pairs.csv \
  --top-k 10
```

## Статус

Это baseline-реализация.

Она сохранена как основа, потому что по сравнению с экспериментальной retrieval + ranking веткой показала лучшее качество.
