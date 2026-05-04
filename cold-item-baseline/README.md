# BPMSoft Cold-Item Hybrid Recommender

Гибридная рекомендательная система для сценария `cold items`, где объединяются:

- `ALS` как коллаборативная модель
- `CatBoostRegressor` как feature-based модель, которая учится приближать `ALS score`

На инференсе система считает оба предикта для пары `(user, item)`:

- `als_score`
- `regressor_score`

Итоговый скор:

```text
final_score = max(als_score, regressor_score)
```

После этого объекты сортируются по `final_score`, и для каждого пользователя выбирается `top-k`.

## Идея проекта

Проблема cold-item рекомендаций в том, что новые или редкие объекты имеют мало взаимодействий, поэтому чистая коллаборативная фильтрация может работать слабо.

В этом проекте используется следующая логика:

1. `ALS` обучается на матрице взаимодействий `user-item`
2. Все объекты делятся на `warm` и `cold`
3. Для warm-пар считается `ALS score`
4. `CatBoostRegressor` учится предсказывать этот `ALS score` по признакам пользователя и объекта
5. На инференсе для каждой пары `(user, item)` считаются оба score, и берётся максимум

Это позволяет:

- использовать collaborative signal от ALS
- использовать признаки пользователей и объектов
- поддерживать cold-item сценарий

## Структура проекта

```text
BPMSoft_Cold_Item/
├── README.md
├── main_train.py
├── main_infer.py
└── src/
    ├── data_loader.py
    ├── preprocessing.py
    ├── split_warm_cold.py
    ├── als_model.py
    ├── ranker_model.py
    ├── feature_builder.py
    ├── hybrid_recommender.py
    ├── train_pipeline.py
    └── inference_pipeline.py
```

## Что находится в каждом файле

### `main_train.py`

CLI-точка входа для обучения модели.

Запускает:

- чтение train CSV
- обучение `HybridRecommender`
- сохранение модели

### `main_infer.py`

CLI-точка входа для инференса.

Запускает:

- загрузку сохранённой модели
- скоринг candidate pairs
- построение итоговых рекомендаций

### `src/data_loader.py`

Отвечает за загрузку входного CSV.

Что делает:

- проверяет существование файла
- проверяет, что это `.csv`
- проверяет обязательные колонки
- выделяет optional user/item feature columns

### `src/preprocessing.py`

Подготавливает данные для обучения и инференса.

Что делает:

- валидирует обязательные колонки
- нормализует `user_id` и `item_id`
- приводит `value` к числу
- заменяет `NaN` в `value` на `0`
- отделяет признаки `user_*` и `item_*`
- строит таблицы user features и item features
- определяет числовые и категориальные признаки
- применяет:
  - `StandardScaler` для числовых признаков
  - `OneHotEncoder` для категориальных признаков

### `src/split_warm_cold.py`

Определяет, какие items считаются `warm`, а какие `cold`.

Поддерживает два режима популярности:

- `count` — число взаимодействий
- `value_sum` — сумма `value`

Основное правило:

```text
if popularity >= threshold -> warm
else -> cold
```

### `src/als_model.py`

Обёртка над библиотечной моделью `implicit.als.AlternatingLeastSquares`.

Что делает:

- строит sparse `user-item` matrix
- создаёт маппинги `user_id <-> index` и `item_id <-> index`
- обучает ALS
- считает `ALS score` для пары `(user, item)`
- строит top-N рекомендации

### `src/ranker_model.py`

Здесь находится вторая модель проекта: `CatBoostRegressor`.

Что делает:

- учится предсказывать `ALS score`
- принимает уже подготовленную feature matrix
- возвращает численный `regressor_score`

Важно:

- несмотря на имя файла `ranker_model.py`, внутри сейчас именно `CatBoostRegressor`
- это соответствует текущей архитектуре проекта

### `src/feature_builder.py`

Готовит обучающий датасет для `CatBoostRegressor`.

Что делает:

- берёт только `warm items`
- формирует observed warm-пары
- при необходимости добавляет sampled negative warm-пары
- строит признаки пары `(user, item)`
- считает target как `ALS score`

### `src/hybrid_recommender.py`

Главный оркестратор всей системы.

Что делает:

- обучает preprocessing
- делает split warm/cold
- обучает ALS
- обучает CatBoostRegressor
- на инференсе считает:
  - `als_score`
  - `regressor_score`
  - `final_score = max(als_score, regressor_score)`

### `src/train_pipeline.py`

Полный pipeline обучения.

Что делает:

- загружает train CSV
- создаёт все компоненты модели
- обучает `HybridRecommender`
- сохраняет модель
- возвращает summary обучения

### `src/inference_pipeline.py`

Полный pipeline инференса.

Что делает:

- загружает сохранённую модель
- загружает candidate pairs CSV
- считает score
- строит top-k рекомендации
- при необходимости сохраняет результаты в CSV

## Формат входных данных

### Обязательные колонки

Train CSV обязательно должен содержать:

- `user_id`
- `item_id`
- `value`

### Опциональные колонки

Поддерживаются:

- `user_*` — признаки пользователя
- `item_*` — признаки объекта

Все остальные колонки считаются дополнительными и в модель не идут автоматически.

### Пример train CSV

```csv
user_id,item_id,value,user_age,user_city,item_category,item_price
u1,i10,1,25,Moscow,Books,500
u1,i11,2,25,Moscow,Electronics,45000
u2,i10,1,31,Kazan,Books,500
u3,i12,3,22,SPB,Home,7800
```

### Пример candidate pairs CSV для инференса

```csv
user_id,item_id
u1,i10
u1,i12
u1,i15
u2,i10
u2,i16
```

Если у тебя есть признаки пользователя и объекта в candidate CSV, они тоже могут быть переданы, но базово для инференса обязательны только:

- `user_id`
- `item_id`

## Как устроен pipeline обучения

### Шаг 1. Загрузка CSV

Используется `src/data_loader.py`.

Проверяется:

- что файл существует
- что это CSV
- что есть `user_id`, `item_id`, `value`

### Шаг 2. Предобработка

Используется `src/preprocessing.py`.

Результат:

- очищенный `interactions_df`
- `user_features_df`
- `item_features_df`
- fitted feature preprocessor

### Шаг 3. Split warm/cold

Используется `src/split_warm_cold.py`.

Все items делятся на:

- `warm_items`
- `cold_items`

### Шаг 4. Обучение ALS

Используется `src/als_model.py`.

ALS учится на взаимодействиях и становится teacher-моделью.

### Шаг 5. Подготовка датасета для CatBoostRegressor

Используется `src/feature_builder.py`.

Для warm-пар:

- строятся признаки
- считается `ALS score`
- `ALS score` становится target для регрессии

### Шаг 6. Обучение CatBoostRegressor

Используется `src/ranker_model.py`.

Модель учится по признакам восстанавливать `ALS score`.

### Шаг 7. Сохранение `HybridRecommender`

Готовая модель сохраняется через `joblib`.

## Как устроен pipeline инференса

### Шаг 1. Загрузка модели

Используется сохранённый `HybridRecommender`.

### Шаг 2. Загрузка candidate pairs

CSV должен содержать минимум:

- `user_id`
- `item_id`

### Шаг 3. Построение признаков

Используется fitted preprocessing из обученной модели.

### Шаг 4. Вычисление двух score

Для каждой пары считаются:

- `als_score`
- `regressor_score`

### Шаг 5. Объединение score

Используется правило:

```text
final_score = max(als_score, regressor_score)
```

### Шаг 6. Построение рекомендаций

Пары сортируются по:

- `user_id` по возрастанию
- `final_score` по убыванию

После этого берётся `top-k` объектов на пользователя.

## Установка зависимостей

Проект использует следующие библиотеки:

- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `joblib`
- `implicit`
- `catboost`

Пример установки:

```bash
pip install pandas numpy scipy scikit-learn joblib implicit catboost
```

## Запуск обучения

Минимальный запуск:

```bash
python main_train.py --train-csv data/train.csv --model-output artifacts/hybrid_model.joblib
```

Пример с дополнительными параметрами:

```bash
python main_train.py \
  --train-csv data/train.csv \
  --model-output artifacts/hybrid_model.joblib \
  --min-warm-interactions 5 \
  --popularity-metric count \
  --als-factors 20 \
  --als-regularization 0.01 \
  --als-iterations 150 \
  --als-alpha 20 \
  --regressor-iterations 300 \
  --regressor-learning-rate 0.05 \
  --regressor-depth 5 \
  --negative-samples-per-user 3
```

После обучения в консоль выводится summary:

- число строк
- число пользователей
- число объектов
- число warm/cold items
- какие user/item features были найдены

## Запуск инференса

Минимальный запуск:

```bash
python main_infer.py \
  --model-path artifacts/hybrid_model.joblib \
  --input-csv data/candidates.csv
```

Пример с output-файлами:

```bash
python main_infer.py \
  --model-path artifacts/hybrid_model.joblib \
  --input-csv data/candidates.csv \
  --top-k 10 \
  --scored-output outputs/scored_pairs.csv \
  --recommendations-output outputs/recommendations.csv
```

## Что возвращает инференс

### `scored_pairs.csv`

Содержит:

- `user_id`
- `item_id`
- признаки пары
- `als_score`
- `regressor_score`
- `item_group`
- `final_score`

### `recommendations.csv`

Содержит top-k рекомендации по каждому пользователю:

- `user_id`
- `item_id`
- `final_score`
- `rank`

## Когда использовать этот проект

Этот проект подходит, если ты хочешь:

- обучать рекомендательную систему на CSV
- использовать как interaction history, так и признаки пользователей и объектов
- работать со сценарием cold items
- получить понятную модульную архитектуру без лишней инфраструктуры

## Краткое резюме архитектуры

Идея проекта в одной цепочке:

```text
CSV
-> data_loader
-> preprocessing
-> split_warm_cold
-> ALS
-> feature_builder
-> CatBoostRegressor
-> HybridRecommender
-> inference
-> max(ALS score, regressor score)
-> top-k recommendations
```

Если ты новый пользователь проекта, то тебе достаточно помнить две команды:

```bash
python main_train.py --train-csv data/train.csv --model-output artifacts/hybrid_model.joblib
python main_infer.py --model-path artifacts/hybrid_model.joblib --input-csv data/candidates.csv
```
