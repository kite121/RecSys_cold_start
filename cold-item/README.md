# BPMSoft Cold-Item Retrieval + Ranking

Рекомендательная система для сценария `cold items`, где retrieval и ranking разделены на два отдельных слоя:

- `ALS` используется как retrieval-модель для `warm items`
- для `cold items` строятся `synthetic latent vectors`
- итоговый candidate pool ранжируется через `CatBoostRanker`

На инференсе система:

- генерирует `warm candidates` через `ALS`
- генерирует `cold candidates` через synthetic vectors
- объединяет кандидатов в один shortlist
- считает `ranker_score`
- возвращает `top-k` рекомендаций

## Идея проекта

Проблема cold-item рекомендаций в том, что новые или редкие объекты почти не имеют interaction history, поэтому чистая collaborative filtering-модель не может оценивать их так же надёжно, как warm items.

В этой реализации используется следующая логика:

1. Все items делятся на `warm` и `cold`
2. `ALS` обучается только на `warm interactions`
3. Из warm items строится support set:
   - `top-N popular`
   - `top-K diverse` через maxvol-style selection
4. Для каждого cold item ищутся ближайшие support neighbors
5. Для каждого cold item строится synthetic latent vector
6. Для каждого пользователя retrieval-слой формирует candidate pool:
   - `als_warm`
   - `cold_vector`
7. `CatBoostRanker` сортирует уже готовый shortlist кандидатов

Это позволяет:

- использовать collaborative signal от ALS там, где он надёжен
- не терять cold items на retrieval-слое
- учить ранкер на realistic candidate pool, а не на всём каталоге

## Структура проекта

```text
cold-item/
├── README.md
├── config.py
├── main_train.py
├── main_infer.py
└── src/
    ├── data_loader.py
    ├── preprocessing.py
    ├── split_warm_cold.py
    ├── als_model.py
    ├── popular_selector.py
    ├── maxvol_selector.py
    ├── similarity_index.py
    ├── cold_vector_builder.py
    ├── candidate_generator.py
    ├── retrieval_model.py
    ├── feature_builder.py
    ├── ranker_model.py
    ├── train_pipeline.py
    ├── inference_pipeline.py
    └── utils.py
```

## Что находится в каждом файле

### `main_train.py`

CLI-точка входа для обучения новой cold-item системы.

Запускает:

- чтение train CSV
- полный train pipeline
- сохранение артефактов в `artifacts/`

### `main_infer.py`

CLI-точка входа для инференса.

Запускает:

- загрузку артефактов из `artifacts/`
- генерацию candidate pool для пользователя
- ranking и возврат `top-k`

### `config.py`

Центральный конфиг новой реализации.

Что хранит:

- пути к артефактам
- настройки CSV-схемы
- параметры split warm/cold
- параметры ALS
- параметры retrieval
- параметры ranker
- параметры inference

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
- отделяет признаки `user_*` и `item_*`
- строит `user_features_df` и `item_features_df`
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

Обёртка над `implicit.als.AlternatingLeastSquares`.

Что делает:

- строит sparse `user-item` matrix
- создаёт маппинги `user_id <-> index` и `item_id <-> index`
- обучает ALS
- считает `ALS score` для пары `(user, item)`
- строит warm recommendations

### `src/popular_selector.py`

Строит `top-N popular` support pool.

Что делает:

- считает popularity score объектов
- умеет использовать:
  - interaction weights
  - time decay
  - user cap
  - bonus за число уникальных пользователей
- возвращает shortlist популярных warm items

### `src/maxvol_selector.py`

Строит `top-K diverse` support set из `top-N popular`.

Что делает:

- проецирует feature matrix в компактное dense space
- нормализует векторы
- применяет `QR pivoting`
- при необходимости добирает элементы greedy-diversity heuristic

### `src/similarity_index.py`

Ищет support neighbors для cold items.

Что делает:

- переводит cold и support items в общее feature space
- считает similarity между cold и support items
- возвращает `top-M nearest neighbors` для каждого cold item

### `src/cold_vector_builder.py`

Строит synthetic latent vectors для cold items.

Что делает:

- берёт найденных support neighbors
- достаёт latent vectors warm items из ALS
- агрегирует соседские векторы
- возвращает `cold_vector_map`

### `src/candidate_generator.py`

Генерирует retrieval-stage candidate pool.

Что делает:

- для warm items использует `ALS`
- для cold items использует `user_vector · cold_vector`
- объединяет кандидатов
- удаляет дубликаты
- назначает:
  - `retrieval_score`
  - `retrieval_source`
  - `retrieval_rank`

### `src/retrieval_model.py`

Оркестратор retrieval-слоя.

Что делает:

- строит общий item feature space
- управляет:
  - `popular_selector`
  - `maxvol_selector`
  - `similarity_index`
  - `cold_vector_builder`
- подготавливает retrieval artifacts

### `src/feature_builder.py`

Готовит датасет для `CatBoostRanker`.

Что делает:

- берёт candidate pairs
- присоединяет user/item features
- добавляет retrieval features:
  - `retrieval_score`
  - `retrieval_rank`
  - `retrieval_source`
  - `is_cold_item`
- формирует:
  - `feature_matrix`
  - `labels`
  - `group_id`

### `src/ranker_model.py`

Финальная ranking-модель проекта: `CatBoostRanker`.

Что делает:

- обучается на candidate pairs
- использует `group_id = user_id`
- считает `ranker_score`
- сортирует кандидатов внутри пользователя

### `src/train_pipeline.py`

Полный pipeline обучения.

Что делает:

- загружает train CSV
- обучает preprocessing
- делает split warm/cold
- обучает ALS
- строит support set
- строит cold vectors
- генерирует training candidates
- строит ranker dataset
- обучает `CatBoostRanker`
- сохраняет артефакты

### `src/inference_pipeline.py`

Полный pipeline инференса.

Что делает:

- загружает train-time артефакты
- генерирует candidate pool для пользователя
- строит ranker features
- считает `ranker_score`
- возвращает итоговые рекомендации

### `src/utils.py`

Вспомогательные функции для работы с артефактами.

Что делает:

- собирает canonical artifact paths
- загружает сохранённые train-time artifacts
- строит summary загруженной модели

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

Все остальные колонки сохраняются в raw dataframe, но в feature pipeline автоматически не участвуют.

### Пример train CSV

```csv
user_id,item_id,value,user_age,user_city,item_category,item_price
u1,i10,1,25,Moscow,Books,500
u1,i11,2,25,Moscow,Electronics,45000
u2,i10,1,31,Kazan,Books,500
u3,i12,3,22,SPB,Home,7800
```

## Что нужно загрузить, чтобы проверить систему

Сейчас для проверки системы нужен один основной CSV:

- `train.csv`

Именно он используется для:

- обучения preprocessing
- split warm/cold
- обучения ALS
- построения support set
- построения cold vectors
- обучения ranker

После этого инференс не требует отдельного CSV с candidate pairs:

- candidate pool генерируется внутри системы автоматически
- пользователю достаточно передать `user_id`

Если нужно проверить cold/warm поведение, лучше всего использовать CSV, где:

- есть несколько пользователей
- есть несколько часто встречающихся items
- есть часть редких items с малым числом взаимодействий
- есть хотя бы несколько `user_*` и `item_*` признаков

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

ALS обучается только на `warm interactions`.

### Шаг 5. Построение support set

Используются:

- `src/popular_selector.py`
- `src/maxvol_selector.py`

Сначала берётся `top-N popular`, затем из него строится `top-K diverse`.

### Шаг 6. Поиск соседей для cold items

Используется `src/similarity_index.py`.

Для каждого cold item находятся ближайшие support neighbors.

### Шаг 7. Построение synthetic cold vectors

Используется `src/cold_vector_builder.py`.

Каждый cold item получает synthetic latent vector.

### Шаг 8. Candidate generation

Используется `src/candidate_generator.py`.

Для train users строится retrieval shortlist.

### Шаг 9. Подготовка ranker dataset

Используется `src/feature_builder.py`.

Для candidate pairs строятся:

- признаки
- labels
- group ids

### Шаг 10. Обучение CatBoostRanker

Используется `src/ranker_model.py`.

Модель учится сортировать shortlist кандидатов внутри пользователя.

### Шаг 11. Сохранение артефактов

Сохраняются:

- `project_config.joblib`
- `preprocessing_artifacts.joblib`
- `preprocessor.joblib`
- `warm_cold_split.joblib`
- `support_items.joblib`
- `cold_neighbors.joblib`
- `cold_vectors.joblib`
- `als_model.joblib`
- `ranker_model.joblib`

## Как устроен pipeline инференса

### Шаг 1. Загрузка артефактов

Система загружает train-time артефакты из `artifacts/`.

### Шаг 2. Генерация кандидатов

Для пользователя строятся:

- warm candidates через `ALS`
- cold candidates через synthetic vectors

### Шаг 3. Построение ranker features

Используется сохранённый preprocessing.

### Шаг 4. Ranking

`CatBoostRanker` считает `ranker_score`.

### Шаг 5. Построение рекомендаций

Кандидаты сортируются по:

- `group_id`
- `ranker_score` по убыванию
- `retrieval_rank` как tie-breaker

После этого берётся `top-k`.

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
python main_train.py --train-csv data/train.csv
```

Пример с дополнительными параметрами:

```bash
python main_train.py \
  --train-csv data/train.csv \
  --artifacts-dir artifacts \
  --min-warm-interactions 5 \
  --popularity-metric count \
  --als-factors 64 \
  --als-regularization 0.01 \
  --als-iterations 100 \
  --als-alpha 20 \
  --top-n-popular 5000 \
  --top-k-diverse 500 \
  --top-m-neighbors 20 \
  --warm-candidates-per-user 200 \
  --cold-candidates-per-user 200 \
  --final-candidate-pool-size 400 \
  --ranker-iterations 300 \
  --ranker-learning-rate 0.05 \
  --ranker-depth 6
```

После обучения в консоль выводится summary:

- число строк
- число пользователей
- число объектов
- число warm/cold items
- число support items
- число cold vectors
- число ranker pairs

## Запуск инференса

Минимальный запуск:

```bash
python main_infer.py --user-id u1
```

Пример с дополнительными параметрами:

```bash
python main_infer.py \
  --user-id u1 \
  --artifacts-dir artifacts \
  --top-k 10 \
  --user-context user_age=25 \
  --user-context user_city=Moscow
```

Важно:

- `main_infer.py` использует сохранённый `project_config.joblib`
- это значит, что inference поднимает те же retrieval/ranker настройки, с которыми система обучалась

## Какие функции вызывать из Python

### Обучение

Главная функция:

```python
from src.train_pipeline import train_cold_item_pipeline

result = train_cold_item_pipeline(
    train_csv_path="data/train.csv",
)
```

Объектный вариант:

```python
from src.train_pipeline import ColdItemTrainPipeline

pipeline = ColdItemTrainPipeline()
result = pipeline.run("data/train.csv")
```

### Инференс

Главная функция:

```python
from src.inference_pipeline import run_cold_item_inference

result = run_cold_item_inference(
    user_id="u1",
    top_k=10,
)
```

Объектный вариант:

```python
from src.inference_pipeline import ColdItemInferencePipeline

pipeline = ColdItemInferencePipeline()
result = pipeline.run(user_id="u1", top_k=10)
```

## Что возвращает инференс

Итоговый `InferencePipelineResult` содержит:

- `candidates_df` — retrieval candidate pool
- `ranker_dataset` — подготовленные ranker features
- `prediction_result` — scored pairs
- `recommendations_df` — итоговые рекомендации
- `inference_summary` — краткую сводку

В `recommendations_df` обычно есть:

- `user_id`
- `item_id`
- `retrieval_score`
- `retrieval_source`
- `retrieval_rank`
- `is_cold_item`
- `ranker_score`

## Когда использовать этот проект

Этот проект подходит, если ты хочешь:

- обучать рекомендательную систему на CSV
- использовать и interaction history, и user/item features
- работать со сценарием `cold items`
- получать top-k рекомендации без ручной подготовки candidate CSV

## Текущие ограничения

На текущем этапе важно понимать:

- для обучения нужен CSV формата `user_id,item_id,value,...`
- inference сейчас работает по `user_id`, а не по отдельному input CSV
- batch inference через CSV candidate pairs пока не реализован
- для полноценной проверки нужно прогнать `train -> artifacts -> infer` на реальном датасете

## Краткое резюме архитектуры

Идея проекта в одной цепочке:

```text
train.csv
-> data_loader
-> preprocessing
-> split_warm_cold
-> ALS (warm only)
-> top-N popular
-> top-K diverse
-> nearest neighbors for cold items
-> synthetic cold vectors
-> candidate generation
-> feature_builder
-> CatBoostRanker
-> artifacts
-> inference by user_id
-> top-k recommendations
```

Если ты новый пользователь проекта, то тебе достаточно помнить две команды:

```bash
python main_train.py --train-csv data/train.csv
python main_infer.py --user-id u1
```
