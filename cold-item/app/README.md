# Cold-item API (FastAPI)

Этот модуль даёт HTTP-доступ к новой реализации `cold-item` пайплайна: обучение (`/train`) и инференс (`/predict`).

## Запуск

```bash
cd cold-item
uv run fastapi dev app/main.py
```

После старта:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Healthcheck: `GET /health`

## Артефакты

`POST /train` сохраняет модель и все нужные артефакты в:

`cold-item/artifacts/<model_id>/`

`POST /predict` загружает `project_config.joblib` и остальные артефакты из этой же папки по `model_id`.

## `POST /train`

Формат запроса: **`multipart/form-data`**.

- **`train_csv`**: CSV-файл
- **`config`** (опционально): JSON-часть с параметрами (контент-тайп `application/json`), структура соответствует `app/schemas.py::TrainConfig`

Пример через `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/train" \
  -F "train_csv=@/path/to/train.csv;type=text/csv" \
  -F "config=@/path/to/config.json;type=application/json"
```

Ответ:

- `model_id` — идентификатор модели (UUID)
- `summary` — краткая статистика обучения и путь к артефактам

## `POST /predict`

Запрос — JSON.

Минимальный пример:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "PUT_MODEL_ID_HERE",
    "user_id": "123",
    "top_k": 10
  }'
```

Ответ содержит:

- `recommendations`: список `{ item_id, score, rank }`
- `summary`: сводка инференса (например, сколько кандидатов рассмотрено)

