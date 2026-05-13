# Cold-User Hybrid Recommender

Это текущая unified-реализация для пользовательского cold-start сценария.

Поддерживаются три режима:

- `warm-user`
- `cold-user`
- `global cold-start`

## Основная идея

Система делит ответственность между несколькими блоками:

- `ALS` используется только там, где collaborative-сигнал надёжен
- `CatBoostRegressor` учится предсказывать `ALS score` по признакам пары `(user, item)`
- для `cold-user` используется fallback:
  - `popular -> maxvol -> regressor`
- если warm-zone нет вообще, используется `global_cold_start`

## Основной inference

Главный вход:

```text
user_id
optional user_features
top_k
```

То есть основной сценарий здесь уже не использует внешний `candidate_pairs.csv`.

## Логика по режимам

### Warm-user

Если пользователь warm:

- `warm items -> ALS`
- `cold items -> RegressorModel`
- затем merge, ranking и `top-k`

### Cold-user

Если пользователь cold или unseen:

- берутся `top-N popular items`
- они сокращаются через `MaxVol`
- затем кандидаты скорятся через `RegressorModel`
- затем возвращается `top-k`

### Global cold-start

Если usable warm collaborative zone нет:

- ALS не обучается
- `popular + maxvol` остаются fallback candidate generation
- если регрессор не обучен, рекомендации ранжируются по popularity

## Структура проекта

```text
cold-user/
├── README.md
├── config.py
├── main_train.py
├── main_infer.py
└── src/
    ├── als_model.py
    ├── cold_user_recommender.py
    ├── data_loader.py
    ├── feature_builder.py
    ├── hybrid_recommender.py
    ├── inference_pipeline.py
    ├── maxvol_selector.py
    ├── popular_selector.py
    ├── preprocessing.py
    ├── regressor_model.py
    ├── split_warm_cold.py
    ├── split_warm_cold_users.py
    ├── train_pipeline.py
    └── utils.py
```

## Что делает каждый модуль

- `config.py`
  - defaults для train/infer параметров и путей

- `main_train.py`
  - CLI для обучения

- `main_infer.py`
  - CLI для inference по `user_id`

- `data_loader.py`
  - загрузка train CSV и определение user/item feature columns

- `preprocessing.py`
  - нормализация ids, построение feature tables и shared preprocessing для пары `(user, item)`

- `split_warm_cold.py`
  - split items на `warm/cold`

- `split_warm_cold_users.py`
  - split users на `warm/cold`

- `als_model.py`
  - ALS wrapper для warm collaborative части

- `popular_selector.py`
  - fitted popularity selector

- `maxvol_selector.py`
  - fitted diversification selector

- `feature_builder.py`
  - train features, negative sampling и inference features

- `regressor_model.py`
  - `CatBoostRegressor`

- `cold_user_recommender.py`
  - отдельный fallback flow:
  - `popular -> maxvol -> regressor`

- `hybrid_recommender.py`
  - главный runtime router между режимами

- `train_pipeline.py`
  - полный orchestration обучения

- `inference_pipeline.py`
  - orchestration inference по `user_id`

## Train pipeline

Текущий порядок обучения:

1. загрузка train CSV
2. preprocessing
3. split users на `warm/cold`
4. split items на `warm/cold`
5. построение `warm_interactions_df`
6. проверка `global_cold_start`
7. fit `PopularSelector`
8. fit `MaxVolSelector`
9. если warm-zone есть:
   - train `ALS`
   - сбор positive/negative train pairs
   - train `RegressorModel`
10. упаковка всего в `HybridRecommender`

## Формат train CSV

Обязательные колонки:

- `user_id`
- `item_id`
- `value`

Опциональные:

- `user_*`
- `item_*`

## CLI

Обучение:

```bash
python cold-user/main_train.py \
  --train-csv path/to/train.csv \
  --model-output cold-user/artifacts/hybrid_model.joblib
```

Инференс:

```bash
python cold-user/main_infer.py \
  --model-path cold-user/artifacts/hybrid_model.joblib \
  --user-id 123 \
  --top-k 10
```

С optional user features:

```bash
python cold-user/main_infer.py \
  --model-path cold-user/artifacts/hybrid_model.joblib \
  --user-id 123 \
  --user-features-json '{"user_age": 25}' \
  --top-k 10
```

## Что проверено

По текущему состоянию кода рабочими прогонялись:

- train в `hybrid` режиме
- inference для `warm-user`
- inference для `cold-user`
- inference для `cold-user` без дополнительных `user_features`
- train в `global_cold_start`
- inference для `global_cold_start`

То есть основной current pipeline в `cold-user/` сейчас работает end-to-end.
