from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(tags=["train"])


@router.post("/train")
async def train_endpoint(
    train_csv: UploadFile = File(..., description="Training interactions CSV."),
) -> dict:
    raise HTTPException(
        status_code=501,
        detail={
            "status": "not_implemented",
            "message": "POST /train is a stub: no training logic wired yet.",
        },
    )

