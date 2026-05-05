from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["predict"])


@router.post("/predict")
def predict_endpoint(body: dict) -> dict:
    raise HTTPException(
        status_code=501,
        detail={
            "status": "not_implemented",
            "message": "POST /predict is a stub: no inference logic wired yet.",
            "request_echo": body,
        },
    )

