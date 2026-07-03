import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .database import engine, get_db
from .models import Base, Detection
from .detector import detect_plate

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Number Plate Detection")

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload JPG or PNG.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        result = detect_plate(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    detection = Detection(
        filename=file.filename,
        plate_text=result["plate_text"],
        confidence=result["confidence"],
        candidates_json=json.dumps(result["candidates"]),
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)

    return {
        "id": detection.id,
        "plate_text": result["plate_text"],
        "confidence": result["confidence"],
        "annotated_b64": result["annotated_b64"],
        "plate_crop_b64": result["plate_crop_b64"],
        "candidates": result["candidates"],
    }


@app.get("/api/history")
async def history(db: Session = Depends(get_db)):
    rows = db.query(Detection).order_by(Detection.created_at.desc()).limit(20).all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "plate_text": r.plate_text,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@app.delete("/api/history")
async def clear_history(db: Session = Depends(get_db)):
    count = db.query(Detection).count()
    db.query(Detection).delete()
    db.commit()
    return {"deleted": count}
