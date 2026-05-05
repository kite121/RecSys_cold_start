from fastapi import FastAPI

from app.api.predict import router as predict_router
from app.api.train import router as train_router

app = FastAPI(title="cold-item API", version="0.0.1")
app.include_router(train_router)
app.include_router(predict_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

