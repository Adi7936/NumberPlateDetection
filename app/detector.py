import base64
import io
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image


def detect_plate(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")

    plate_region, bbox = _locate_plate(img)

    candidates = []

    # OCR on plate region
    if plate_region is not None:
        candidates += _ocr_region(plate_region)

    # Always run OCR on full image as fallback
    full_candidates = _ocr_region(img, full_image=True)
    for c in full_candidates:
        if not any(x["text"] == c["text"] for x in candidates):
            candidates.append(c)

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    best = _pick_best(candidates)

    plate_text = best["text"] if best else "Not detected"
    confidence = best["confidence"] if best else 0.0

    # Draw bounding box
    annotated = img.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (78, 204, 163), 3)
        label_bg_w = max(len(plate_text) * 16 + 10, w)
        cv2.rectangle(annotated, (x, y - 36), (x + label_bg_w, y), (78, 204, 163), -1)
        cv2.putText(annotated, plate_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 17, 23), 2)

    return {
        "plate_text": plate_text,
        "confidence": confidence,
        "annotated_b64": _img_to_b64(annotated),
        "plate_crop_b64": _img_to_b64(plate_region) if plate_region is not None else None,
        "candidates": candidates[:5],
    }


def _ocr_region(img: np.ndarray, full_image: bool = False) -> list[dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    results = []
    configs = [
        "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        "--psm 6",
    ]

    seen = set()
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
            for i, text in enumerate(data["text"]):
                cleaned = _clean_plate(text)
                conf_raw = int(data["conf"][i])
                if cleaned and conf_raw > 0 and cleaned not in seen:
                    seen.add(cleaned)
                    # Normalize confidence to 0-1
                    conf = conf_raw / 100.0
                    if full_image and conf < 0.5:
                        continue
                    results.append({"text": cleaned, "confidence": round(conf, 3)})
        except Exception:
            continue

    return results


def _locate_plate(img: np.ndarray):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    # Morphological fallback
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

    # Haar cascade fallback
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    try:
        cascade = cv2.CascadeClassifier(cascade_path)
        plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
        if len(plates) > 0:
            x, y, cw, ch = plates[0]
            return img[y:y + ch, x:x + cw], (x, y, cw, ch)
    except Exception:
        pass

    return None, None


def _pick_best(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    plate_pattern = re.compile(r'^[A-Z0-9]{4,12}$')
    plate_candidates = [c for c in candidates if plate_pattern.match(c["text"])]
    return plate_candidates[0] if plate_candidates else candidates[0]


def _clean_plate(text: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9\-]", "", text.upper().strip())
    return cleaned if len(cleaned) >= 3 else ""


def _img_to_b64(img: np.ndarray) -> str:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
