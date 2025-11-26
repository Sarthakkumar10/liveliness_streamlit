# ============================================================
# main.py — FULL (UPDATED FOR PRODUCTION)
# - Accepts multipart file OR JSON {"image_base64": "..."}
# - Removes processed_image_base64 from responses
# - All thresholds and paths configured via environment variables
# - Improved error handling and logging
# - Safe defaults so server won't crash in absence of env
# ============================================================

import io
import os
import cv2
import time
import math
import json
import base64
import traceback
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests

# ML libraries
from ultralytics import YOLO
import mediapipe as mp

# SCRFD import (best-effort)
try:
    from insightface.model_zoo.scrfd import SCRFD
except Exception:
    try:
        from scrfd import SCRFD
    except Exception as e:
        SCRFD = None


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("face_phone_api")


# ============================================================
# ENV / CONFIG
# ============================================================
# model paths
MODEL_YOLO_PATH = os.getenv("MODEL_YOLO_PATH", "best_960.pt")
SCRFD_PATH = os.getenv("SCRFD_PATH", "scrfd.onnx")

# thresholds
YAW_THRESH = float(os.getenv("YAW_THRESH", 34))
PITCH_THRESH = float(os.getenv("PITCH_THRESH", 31))
ROLL_THRESH = float(os.getenv("ROLL_THRESH", 30))

BRIGHT_LOW = float(os.getenv("BRIGHT_LOW", 50))
BRIGHT_HIGH = float(os.getenv("BRIGHT_HIGH", 150))

BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 50.0))

PHONE_CONFIDENCE_THRESHOLD = float(os.getenv("PHONE_CONFIDENCE_THRESHOLD", 0.45))

# neuroverify
NEUROVERIFY_URL = os.getenv("NEUROVERIFY_URL", "https://neuroverify.neuraldefend.com/detect/liveness-image")
NEUROVERIFY_APIKEY = os.getenv("NEUROVERIFY_APIKEY", "trulymadly569NVB59NFBN9nfvTESTING")
NEUROVERIFY_TIMEOUT = int(os.getenv("NEUROVERIFY_TIMEOUT", 15))

# save dirs
SAVE_ROOT = os.getenv("SAVE_ROOT", "saved")
VALID_DIR = os.path.join(SAVE_ROOT, "valid")
INVALID_DIR = os.path.join(SAVE_ROOT, "invalid")
PHONE_DIR = os.path.join(SAVE_ROOT, "phone_detected")

os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(INVALID_DIR, exist_ok=True)
os.makedirs(PHONE_DIR, exist_ok=True)

# image encode quality
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 85))


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Face + Phone Validation API (Production)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RESPONSE / INPUT MODELS
# ============================================================
class AnalyzeResult(BaseModel):
    final_valid: bool
    reasons: Dict[str, Any]
    brightness: Optional[float]
    blurriness: Optional[float]
    pose: Optional[Tuple[Optional[float], Optional[float], Optional[float]]]
    num_faces: int
    phone_detected: bool
    phone_confidence: Optional[float]
    spoof_status: Optional[str] = None
    spoof_confidence: Optional[float] = None


class Base64Image(BaseModel):
    image_base64: str


# ============================================================
# SMALL UTILITIES
# ============================================================

def imencode_to_base64(img_bgr) -> str:
    """Convert image to base64 JPG."""
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def compute_brightness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_blurriness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rotation_matrix_to_angles(R):
    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    z = math.atan2(R[1, 0], R[0, 0])
    return np.degrees([x, y, z]).tolist()


def save_processed_image(img_bgr, final_valid, phone_detected, prefix="img"):
    ts = int(time.time())

    if phone_detected:
        save_path = os.path.join(PHONE_DIR, f"{prefix}_{ts}.jpg")
    elif final_valid:
        save_path = os.path.join(VALID_DIR, f"{prefix}_{ts}.jpg")
    else:
        save_path = os.path.join(INVALID_DIR, f"{prefix}_{ts}.jpg")

    try:
        cv2.imwrite(save_path, img_bgr)
    except Exception:
        logger.exception("Failed to write processed image to disk")
        save_path = None

    return save_path


# ============================================================
# External API: NeuroVerify (Liveness)
# ============================================================

def run_neuroverify_api(image_bytes: bytes):
    """Call external liveness API. Returns (status_str, confidence_float).
    Status_str in {"REAL","SPOOF","ERROR"}
    """
    if not NEUROVERIFY_URL or not NEUROVERIFY_APIKEY:
        logger.warning("NeuroVerify config missing, skipping liveness call")
        return "ERROR", 0.0

    headers = {"x-api-key": NEUROVERIFY_APIKEY}
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

    try:
        resp = requests.post(NEUROVERIFY_URL, headers=headers, files=files, timeout=NEUROVERIFY_TIMEOUT)
        logger.debug("NeuroVerify status: %s", resp.status_code)

        if resp.status_code != 200:
            logger.warning("NeuroVerify returned non-200: %s", resp.text)
            return "ERROR", 0.0

        data = resp.json()
        # defensively parse
        tag = data.get("image_analysis", {}).get("prediction_tag")
        conf = data.get("image_analysis", {}).get("liveness_check", {}).get("confidence", 0.0)

        if not tag:
            return "ERROR", float(conf or 0.0)

        if str(tag).upper() == "SPOOF":
            return "SPOOF", float(conf or 0.0)
        else:
            return "REAL", float(conf or 0.0)

    except Exception as e:
        logger.exception("NeuroVerify request failed: %s", e)
        return "ERROR", 0.0


# ============================================================
# MODEL LOADING (once)
# ============================================================

yolo = None
detector = None
mp_face_mesh = mp.solutions.face_mesh


def try_load_models():
    global yolo, detector

    # YOLO
    try:
        logger.info("Loading YOLO model from %s", MODEL_YOLO_PATH)
        yolo = YOLO(MODEL_YOLO_PATH)
        logger.info("YOLO loaded")
    except Exception:
        logger.exception("Failed to load YOLO model")
        yolo = None

    # SCRFD
    if SCRFD is not None:
        try:
            logger.info("Loading SCRFD from %s", SCRFD_PATH)
            detector = SCRFD(SCRFD_PATH)
            ctx_id = 0 if "CUDAExecutionProvider" in detector.session.get_providers() else -1
            detector.prepare(ctx_id, input_size=(640, 640))
            logger.info("SCRFD loaded")
        except Exception:
            logger.exception("Failed to initialize SCRFD detector")
            detector = None
    else:
        logger.warning("SCRFD package not available; face detection will fail")
        detector = None


try_load_models()


# ============================================================
# FACE + POSE + PHONE UTILITIES
# ============================================================

def detect_faces_scrfd(img_bgr):
    if detector is None:
        raise RuntimeError("SCRFD detector not loaded")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bboxes, _ = detector.detect(img_rgb)
    return bboxes  # list of [x1,y1,x2,y2,score]


def pose_from_full_image(img_bgr):
    h, w, _ = img_bgr.shape

    with mp_face_mesh.FaceMesh(static_image_mode=True) as mesh:
        res = mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        return (None, None, None), False

    lm = res.multi_face_landmarks[0]

    # indices chosen as stable landmarks
    idxs = [1, 9, 57, 130, 287, 359]

    pts = []
    for i in idxs:
        try:
            pts.append([int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)])
        except Exception:
            return (None, None, None), False

    model_pts = np.array([
        [285, 528, 200], [285, 371, 152],
        [197, 574, 128], [173, 425, 108],
        [360, 574, 128], [391, 425, 108]
    ], dtype=np.float64)

    img_pts = np.array(pts, dtype=np.float64)

    cam = np.array([[w, 0, w/2],
                    [0, w, h/2],
                    [0, 0, 1]])

    dist = np.zeros((4, 1))

    try:
        ok, rot_vec, _ = cv2.solvePnP(model_pts, img_pts, cam, dist)
    except Exception:
        logger.exception("solvePnP failed")
        return (None, None, None), False

    if not ok:
        return (None, None, None), False

    R, _ = cv2.Rodrigues(rot_vec)
    pitch, yaw, roll = rotation_matrix_to_angles(R)

    frontal = (
        abs(yaw) <= YAW_THRESH and
        abs(pitch) <= PITCH_THRESH and
        abs(roll) <= ROLL_THRESH
    )

    return (pitch, yaw, roll), frontal


def draw_boxes_and_yolo_plot(img_bgr, face_bbox=None, expanded_bbox=None, yolo_results=None):
    if yolo_results is not None and hasattr(yolo_results[0], "plot"):
        try:
            out = yolo_results[0].plot().copy()
        except Exception:
            out = img_bgr.copy()
    else:
        out = img_bgr.copy()

    if face_bbox:
        x1, y1, x2, y2 = map(int, face_bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if expanded_bbox:
        ex1, ey1, ex2, ey2 = map(int, expanded_bbox)
        cv2.rectangle(out, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)

    return out


# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/analyze_frame", response_model=AnalyzeResult)
async def analyze_frame(request: Request, file: UploadFile = File(None)):

    """Accepts either multipart file `file` OR JSON body {"image_base64": "..."}.
    Returns analysis without embedding the processed image.
    """
    start_t = time.time()
    reasons: Dict[str, Any] = {}

    try:
        # ----------------- Load image -----------------
        img = None

        # 1. File upload
        if file is not None:
            try:
                data = await file.read()
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("cv2.imdecode returned None for uploaded file")
            except Exception as e:
                logger.exception("Failed to decode uploaded file: %s", e)
                raise HTTPException(status_code=400, detail=f"Invalid uploaded file: {e}")

        else:
            # 2. JSON Base64
            try:
                raw_body = await request.body()
                logger.info(f"RAW BODY: {raw_body[:500]}")

                if not raw_body:
                    raise HTTPException(status_code=400, detail="Empty JSON body")

                body_json = json.loads(raw_body)

                if "image_base64" not in body_json:
                    raise HTTPException(status_code=400, detail="JSON must contain key 'image_base64'")

                raw = body_json["image_base64"]

                # Remove data URL prefix if present
                if raw.startswith("data:") and ";base64," in raw:
                    raw = raw.split(";base64,", 1)[1]

                decoded = base64.b64decode(raw)
                arr = np.frombuffer(decoded, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("cv2.imdecode returned None for base64")

            except json.JSONDecodeError:
                logger.error("JSON parsing failed. Raw body was not valid JSON.")
                raise HTTPException(status_code=400, detail="Invalid JSON body")

            except Exception as e:
                logger.exception("Base64 JSON decode failed: %s", e)
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


        # 3. If still no image
        if img is None:
            raise HTTPException(status_code=400, detail="Send either multipart file 'file' or JSON body {\"image_base64\":\"...\"}")

        H, W, _ = img.shape

        # ----------------- Face detection -----------------
        try:
            bboxes = detect_faces_scrfd(img)
        except Exception as e:
            logger.exception("Face detection failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Face detector error: {e}")

        num_faces = len(bboxes)
        reasons["num_faces"] = num_faces

        # early returns
        if num_faces == 0:
            reasons["face"] = "no_face"
            save_processed_image(img, final_valid=False, phone_detected=False, prefix="invalid_no_face")
            return AnalyzeResult(
                final_valid=False,
                reasons=reasons,
                brightness=None,
                blurriness=None,
                pose=None,
                num_faces=0,
                phone_detected=False,
                phone_confidence=None,
                spoof_status=None,
                spoof_confidence=None,
            )

        if num_faces > 1:
            reasons["face"] = "multiple_faces"
            save_processed_image(img, final_valid=False, phone_detected=False, prefix="invalid_multiple")
            return AnalyzeResult(
                final_valid=False,
                reasons=reasons,
                brightness=None,
                blurriness=None,
                pose=None,
                num_faces=num_faces,
                phone_detected=False,
                phone_confidence=None,
                spoof_status=None,
                spoof_confidence=None,
            )

        # single face
        x1, y1, x2, y2, score = bboxes[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        reasons["face_bbox"] = [x1, y1, x2, y2, float(score)]

        # 5% border check
        margin_x5 = int(0.05 * W)
        margin_y5 = int(0.05 * H)
        reasons["5pct_margin_px"] = [margin_x5, margin_y5]

        if x1 <= margin_x5 or y1 <= margin_y5 or x2 >= (W - margin_x5) or y2 >= (H - margin_y5):
            reasons["face"] = "near_border_5pct"
            save_processed_image(draw_boxes_and_yolo_plot(img, (x1, y1, x2, y2)), final_valid=False, phone_detected=False, prefix="invalid_border")
            return AnalyzeResult(
                final_valid=False,
                reasons=reasons,
                brightness=None,
                blurriness=None,
                pose=None,
                num_faces=num_faces,
                phone_detected=False,
                phone_confidence=None,
                spoof_status=None,
                spoof_confidence=None,
            )

        # expanded crop (10% of face dims)
        fw, fh = x2 - x1, y2 - y1
        margin_x10 = int(fw * 0.05)
        margin_y10 = int(fh * 0.05)

        ex1 = max(0, x1 - margin_x10)
        ey1 = max(0, y1 - margin_y10)
        ex2 = min(W, x2 + margin_x10)
        ey2 = min(H, y2 + margin_y10)

        reasons["expanded_bbox"] = [ex1, ey1, ex2, ey2]

        crop = img[ey1:ey2, ex1:ex2]

        if crop.size == 0:
            reasons["face"] = "expanded_crop_invalid"
            save_processed_image(draw_boxes_and_yolo_plot(img, (x1, y1, x2, y2), (ex1, ey1, ex2, ey2)), final_valid=False, phone_detected=False, prefix="invalid_crop")
            return AnalyzeResult(
                final_valid=False,
                reasons=reasons,
                brightness=None,
                blurriness=None,
                pose=None,
                num_faces=num_faces,
                phone_detected=False,
                phone_confidence=None,
                spoof_status=None,
                spoof_confidence=None,
            )

        # brightness
        brightness = compute_brightness(crop)
        reasons["brightness"] = brightness

        if not (BRIGHT_LOW <= brightness <= BRIGHT_HIGH):
            reasons["face"] = "bad_lighting"
            save_processed_image(draw_boxes_and_yolo_plot(img, (x1, y1, x2, y2), (ex1, ey1, ex2, ey2)), final_valid=False, phone_detected=False, prefix="invalid_light")
            return AnalyzeResult(
                final_valid=False,
                reasons=reasons,
                brightness=brightness,
                blurriness=None,
                pose=None,
                num_faces=num_faces,
                phone_detected=False,
                phone_confidence=None,
                spoof_status=None,
                spoof_confidence=None,
            )

        # blurriness
        blurriness = compute_blurriness(crop)
        reasons["blurriness"] = blurriness

        if blurriness < BLUR_THRESHOLD:
            reasons["face"] = "blurry"
            save_processed_image(draw_boxes_and_yolo_plot(img, (x1, y1, x2, y2), (ex1, ey1, ex2, ey2)), final_valid=False, phone_detected=False, prefix="invalid_blur")
            return AnalyzeResult(
                final_valid=False,
                reasons=reasons,
                brightness=brightness,
                blurriness=blurriness,
                pose=None,
                num_faces=num_faces,
                phone_detected=False,
                phone_confidence=None,
                spoof_status=None,
                spoof_confidence=None,
            )

        # pose
        (pitch, yaw, roll), frontal = pose_from_full_image(img)
        reasons["pose"] = {"pitch": pitch, "yaw": yaw, "roll": roll, "frontal": frontal}

        if (pitch is None) or (yaw is None) or (roll is None):
            reasons["face"] = "pose_detection_failed"
            save_processed_image(draw_boxes_and_yolo_plot(img, (x1, y1, x2, y2), (ex1, ey1, ex2, ey2)), final_valid=False, phone_detected=False, prefix="invalid_pose")
            return AnalyzeResult(
                final_valid=False,
                reasons=reasons,
                brightness=brightness,
                blurriness=blurriness,
                pose=(pitch, yaw, roll),
                num_faces=num_faces,
                phone_detected=False,
                phone_confidence=None,
                spoof_status=None,
                spoof_confidence=None,
            )

        if not frontal:
            failed = []
            if abs(yaw) > YAW_THRESH:
                failed.append("yaw_exceeded")
            if abs(pitch) > PITCH_THRESH:
                failed.append("pitch_exceeded")
            if abs(roll) > ROLL_THRESH:
                failed.append("roll_exceeded")

            reasons["face"] = "bad_pose:" + ",".join(failed)
            save_processed_image(draw_boxes_and_yolo_plot(img, (x1, y1, x2, y2), (ex1, ey1, ex2, ey2)), final_valid=False, phone_detected=False, prefix="invalid_pose_limits")
            return AnalyzeResult(
                final_valid=False,
                reasons=reasons,
                brightness=brightness,
                blurriness=blurriness,
                pose=(pitch, yaw, roll),
                num_faces=num_faces,
                phone_detected=False,
                phone_confidence=None,
                spoof_status=None,
                spoof_confidence=None,
            )

        # YOLO phone detection
        phone_detected = False
        phone_conf = 0.0

        if yolo is None:
            logger.warning("YOLO model not loaded; skipping phone detection")
        else:
            try:
                results = yolo(img, verbose=False)
                for box in results[0].boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    label = yolo.names[cls].lower()

                    if "phone" in label:
                        phone_conf = max(phone_conf, conf)
                        if conf >= PHONE_CONFIDENCE_THRESHOLD:
                            phone_detected = True
            except Exception:
                logger.exception("YOLO phone detection failed; continuing without phone detection")

        reasons["phone_detected"] = phone_detected
        reasons["phone_confidence"] = phone_conf

        final_valid = (not phone_detected)

        # spoof check if no phone
        spoof_status = "SKIPPED"
        spoof_conf = None

        if not phone_detected:
            try:
                _, full_buf = cv2.imencode('.jpg', img)
                full_bytes = full_buf.tobytes()
                spoof_status, spoof_conf = run_neuroverify_api(full_bytes)

                if spoof_status == "ERROR":
                    spoof_status = "REAL"
                    spoof_conf = 0.0

                if spoof_status == "SPOOF":
                    final_valid = False
                    reasons["liveness"] = "spoof_detected"
                else:
                    final_valid = final_valid and True
            except Exception:
                logger.exception("Liveness check failed; allowing as REAL by default")
                spoof_status = "ERROR"
                spoof_conf = 0.0

        # save processed visual for debugging/records (but not returned)
        try:
            proc_img = draw_boxes_and_yolo_plot(img, (x1, y1, x2, y2), (ex1, ey1, ex2, ey2), results if 'results' in locals() else None)
            save_processed_image(proc_img, final_valid=final_valid, phone_detected=phone_detected, prefix="processed")
        except Exception:
            logger.exception("Failed to draw/save processed image")

        duration = time.time() - start_t
        logger.info("Processed frame in %.3fs - final_valid=%s phone=%s", duration, final_valid, phone_detected)

        return AnalyzeResult(
            final_valid=final_valid,
            reasons=reasons,
            brightness=brightness,
            blurriness=blurriness,
            pose=(pitch, yaw, roll),
            num_faces=num_faces,
            phone_detected=phone_detected,
            phone_confidence=phone_conf if phone_detected else None,
            spoof_status=spoof_status,
            spoof_confidence=spoof_conf,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /analyze_frame: %s", e)
        # return a generic 500 with helpful message
        raise HTTPException(status_code=500, detail="Internal server error - check logs for details")


# health
@app.get("/health")
async def health():
    status = {
        "status": "ok",
        "models": {
            "yolo_loaded": bool(yolo is not None),
            "scrfd_loaded": bool(detector is not None),
        },
    }
    return status


# ============================================================
# RUN SERVER (when executed directly)
# ============================================================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting server on %s:%s", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
