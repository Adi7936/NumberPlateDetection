import base64
import io
import re

import cv2
import easyocr
import numpy as np
from PIL import Image

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def detect_plate(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")

    reader = get_reader()

    # Try to find plate region first
    plate_region, bbox = _locate_plate(img)

    candidates = []

    # OCR on detected region
    if plate_region is not None:
        results = reader.readtext(plate_region, detail=1, paragraph=False)
        for (_, text, conf) in results:
            cleaned = _clean_plate(text)
            if cleaned:
                candidates.append({"text": cleaned, "confidence": round(conf, 3)})

    # Always also run OCR on full image as fallback / supplement
    full_results = reader.readtext(img, detail=1, paragraph=False)
    for (_, text, conf) in full_results:
        cleaned = _clean_plate(text)
        if cleaned and conf > 0.3:
            # Avoid duplicates
            if not any(c["text"] == cleaned for c in candidates):
                candidates.append({"text": cleaned, "confidence": round(conf, 3)})

    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    # Pick best plate-like candidate (prefer patterns that look like plates)
    best = _pick_best(candidates)
    plate_text = best["text"] if best else "Not detected"
    confidence = best["confidence"] if best else 0.0

    # Draw on image
    annotated = img.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (78, 204, 163), 3)
        label = plate_text
        label_bg_w = max(len(label) * 16 + 10, w)
        cv2.rectangle(annotated, (x, y - 36), (x + label_bg_w, y), (78, 204, 163), -1)
        cv2.putText(annotated, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 17, 23), 2)

    annotated_b64 = _img_to_b64(annotated)
    plate_crop_b64 = _img_to_b64(plate_region) if plate_region is not None else None

    return {
        "plate_text": plate_text,
        "confidence": confidence,
        "annotated_b64": annotated_b64,
        "plate_crop_b64": plate_crop_b64,
        "candidates": candidates[:5],
    }


def _locate_plate(img: np.ndarray):
    """Try multiple strategies to find the plate region."""
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Strategy 1: bilateral filter + Canny contours
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            aspect = cw / float(ch) if ch > 0 else 0
            area_ratio = (cw * ch) / (w * h)
            if 1.5 <= aspect <= 6.5 and cw > 60 and ch > 15 and 0.005 < area_ratio < 0.4:
                return img[y:y + ch, x:x + cw], (x, y, cw, ch)

    # Strategy 2: morphological approach
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kern)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kern)
    contours2, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:20]

    for c in contours2:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch) if ch > 0 else 0
        if 2.0 <= aspect <= 6.0 and cw > 80 and ch > 15:
            return img[y:y + ch, x:x + cw], (x, y, cw, ch)

    # Strategy 3: Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    try:
        cascade = cv2.CascadeClassifier(cascade_path)
        plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
        if len(plates):
            x, y, cw, ch = plates[0]
            return img[y:y + ch, x:x + cw], (x, y, cw, ch)
    except Exception:
        pass

    return None, None


def _pick_best(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None

    # Prefer candidates that match typical plate patterns (mix of letters + numbers, 4-10 chars)
    plate_pattern = re.compile(r'^[A-Z0-9]{4,12}$')
    plate_candidates = [c for c in candidates if plate_pattern.match(c["text"])]

    if plate_candidates:
        return plate_candidates[0]

    # Fallback: return highest confidence
    return candidates[0] if candidates else None


def _clean_plate(text: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9\-]", "", text.upper().strip())
    if len(cleaned) < 3:
        return ""
    return cleaned


def _img_to_b64(img: np.ndarray) -> str:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
