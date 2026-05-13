# Cold-Item Ranker-Only Experiment

Это экспериментальная ветка для сценария `cold-item`, в которой retrieval и ranking разделены на два слоя.

Она сохранена в репозитории как исследовательская версия и как объяснение того, почему основной базой проекта остался baseline из `cold-item/`.

## Идея

В этой реализации использовалась более сложная схема:

- `ALS` только для warm-item retrieval
- `popular + maxvol` для построения support pool
- поиск support neighbors для cold items
- synthetic latent vectors для cold items
- общий candidate pool
- `CatBoostRanker` для финального ранжирования

То есть логика была такой:

```text
warm retrieval + cold retrieval -> candidate pool -> CatBoostRanker
```

## Что здесь есть

```text
cold-item-ranker-only-experiment/
├── README.md
├── config.py
├── main_train.py
├── main_infer.py
├── artifacts_test/
└── src/
    ├── als_model.py
    ├── candidate_generator.py
    ├── cold_vector_builder.py
    ├── data_loader.py
    ├── feature_builder.py
    ├── inference_pipeline.py
    ├── maxvol_selector.py
    ├── popular_selector.py
    ├── preprocessing.py
    ├── ranker_model.py
    ├── retrieval_model.py
    ├── similarity_index.py
    ├── split_warm_cold.py
    ├── train_pipeline.py
    └── utils.py
```

## Основные модули

- `popular_selector.py`
  - строит `top-N popular` support pool

- `maxvol_selector.py`
  - делает diversification support set

- `similarity_index.py`
  - ищет ближайших support neighbors для cold items

- `cold_vector_builder.py`
  - строит synthetic vectors для cold items

- `candidate_generator.py`
  - объединяет warm и cold retrieval candidates

- `retrieval_model.py`
  - оркестратор retrieval слоя

- `ranker_model.py`
  - `CatBoostRanker`

- `train_pipeline.py`
  - полный train pipeline retrieval + ranking системы

- `inference_pipeline.py`
  - inference по `user_id`

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
python cold-item-ranker-only-experiment/main_train.py \
  --train-csv path/to/train.csv \
  --artifacts-dir artifacts
```

Инференс:

```bash
python cold-item-ranker-only-experiment/main_infer.py \
  --user-id 123 \
  --artifacts-dir artifacts \
  --top-k 10
```

## Статус

Эта версия не является текущей основной.

Она была оставлена в репозитории потому что:

- по ней нужны полные зафиксированные эксперименты
- она показывает, какие гипотезы были проверены
- часть её кода оказалась полезной и была переиспользована дальше

В частности, в `cold-user/` из неё были перенесены идеи и реализация для:

- `popular_selector`
- `maxvol_selector`
