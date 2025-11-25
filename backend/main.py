# ============================================================
# main.py — PART 1 / 3
# ============================================================

import io
import os
import cv2
import time
import math
import json
import base64
import traceback
import numpy as np
from typing import Optional, Tuple, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ML libraries
from ultralytics import YOLO
import mediapipe as mp
import requests

# SCRFD import
try:
    from insightface.model_zoo.scrfd import SCRFD
except:
    try:
        from scrfd import SCRFD
    except Exception as e:
        raise RuntimeError("SCRFD import failed. Install `insightface` or `scrfd`.") from e


# ============================================================
# SAVE DIRECTORIES
# ============================================================

SAVE_ROOT = "saved"
VALID_DIR = os.path.join(SAVE_ROOT, "valid")
INVALID_DIR = os.path.join(SAVE_ROOT, "invalid")
PHONE_DIR = os.path.join(SAVE_ROOT, "phone_detected")

os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(INVALID_DIR, exist_ok=True)
os.makedirs(PHONE_DIR, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

MODEL_YOLO_PATH = "best_960.pt"
SCRFD_PATH = "scrfd.onnx"

YAW_THRESH = 34
PITCH_THRESH = 31
ROLL_THRESH = 30

BRIGHT_LOW = 50
BRIGHT_HIGH = 200

BLUR_THRESHOLD = 50.0


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Face + Phone Validation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RESPONSE MODEL
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
    processed_image_base64: Optional[str]
    spoof_status: Optional[str] = None
    spoof_confidence: Optional[float] = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def imencode_to_base64(img_bgr) -> str:
    """Convert image to base64 JPG."""
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def compute_brightness(img_bgr):
    """Average grayscale brightness."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_blurriness(img_bgr):
    """Laplacian variance."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rotation_matrix_to_angles(R):
    """Convert rotation matrix to pitch/yaw/roll."""
    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    z = math.atan2(R[1, 0], R[0, 0])
    return np.degrees([x, y, z]).tolist()


def save_processed_image(img_bgr, final_valid, phone_detected, prefix="img"):
    """Save processed output image into categorized folders."""
    ts = int(time.time())

    if phone_detected:
        save_path = os.path.join(PHONE_DIR, f"{prefix}_{ts}.jpg")
    elif final_valid:
        save_path = os.path.join(VALID_DIR, f"{prefix}_{ts}.jpg")
    else:
        save_path = os.path.join(INVALID_DIR, f"{prefix}_{ts}.jpg")

    cv2.imwrite(save_path, img_bgr)
    return save_path

def run_neuroverify_api(image_bytes: bytes):
    print("[backend] Running NeuroVerify API for spoof detection...")
    url = "https://neuroverify.neuraldefend.com/detect/liveness-image"
    headers = {
        "x-api-key": "trulymadly569NVB59NFBN9nfvTESTING"
    }

    files = {
        "file": ("image.jpg", image_bytes, "image/jpeg")
    }

    try:
        response = requests.post(url, headers=headers, files=files, timeout=15)
        print(f"[backend] NeuroVerify response status: {response.text}")

        if response.status_code != 200:
            return "ERROR", 0.0

        data = response.json()
        tag = data["image_analysis"]["prediction_tag"]
        confidence = data["image_analysis"]["liveness_check"]["confidence"]

        if tag.upper() == "SPOOF":
            return "SPOOF", confidence
        else:
            return "REAL", confidence

    except Exception:
        return "ERROR", 0.0


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("[backend] Loading YOLO model...")
yolo = YOLO(MODEL_YOLO_PATH)
print("[backend] YOLO loaded.")

print("[backend] Loading SCRFD face detector...")
detector = SCRFD(SCRFD_PATH)
ctx_id = 0 if "CUDAExecutionProvider" in detector.session.get_providers() else -1
detector.prepare(ctx_id, input_size=(640, 640))
print("[backend] SCRFD loaded.")

mp_face_mesh = mp.solutions.face_mesh

# ============================================================
# PART 2 / 3 — Core Face & Phone Processing Functions
# ============================================================


# ------------------------------------------------------------
# SCRFD face detection
# ------------------------------------------------------------
def detect_faces_scrfd(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bboxes, _ = detector.detect(img_rgb)
    return bboxes  # each box is [x1, y1, x2, y2, score]


# ------------------------------------------------------------
# Head pose estimation (full image)
# ------------------------------------------------------------
def pose_from_full_image(img_bgr):
    h, w, _ = img_bgr.shape

    with mp_face_mesh.FaceMesh(static_image_mode=True) as mesh:
        res = mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        return (None, None, None), False

    lm = res.multi_face_landmarks[0]

    # good points for PnP
    idxs = [1, 9, 57, 130, 287, 359]

    pts = []
    for i in idxs:
        pts.append([int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)])

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

    ok, rot_vec, _ = cv2.solvePnP(model_pts, img_pts, cam, dist)

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


# ------------------------------------------------------------
# YOLO + face boxes visualizer
# ------------------------------------------------------------
def draw_boxes_and_yolo_plot(img_bgr, face_bbox=None, expanded_bbox=None, yolo_results=None):
    """Draw YOLO results + green face box + yellow expanded box."""

    if yolo_results is not None:
        out = yolo_results[0].plot().copy()
    else:
        out = img_bgr.copy()

    # green face box
    if face_bbox:
        x1, y1, x2, y2 = map(int, face_bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # yellow expanded box
    if expanded_bbox:
        ex1, ey1, ex2, ey2 = map(int, expanded_bbox)
        cv2.rectangle(out, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)

    return out

# ============================================================
# PART 3 / 3 — API Endpoint + Saving Logic + Run Server
# ============================================================


# ------------------------------------------------------------
# Helper: Save processed image into proper folder
# ------------------------------------------------------------
def save_processed_image(img_bgr, final_valid, phone_detected, prefix="img"):
    """
    Save processed output into:
        saved/valid/
        saved/invalid/
        saved/phone_detected/
    """

    ts = int(time.time())

    if phone_detected:
        save_path = os.path.join(PHONE_DIR, f"{prefix}_{ts}.jpg")
    elif final_valid:
        save_path = os.path.join(VALID_DIR, f"{prefix}_{ts}.jpg")
    else:
        save_path = os.path.join(INVALID_DIR, f"{prefix}_{ts}.jpg")

    cv2.imwrite(save_path, img_bgr)
    return save_path


# ------------------------------------------------------------
# API: Upload Frame → Validate Face → Check Phone
# ------------------------------------------------------------
@app.post("/analyze_frame", response_model=AnalyzeResult)
async def analyze_frame(file: UploadFile = File(...)):
    """
    Accepts an uploaded image.
    Runs:
        - SCRFD detection
        - 5% border rule
        - 10% expanded brightness/blur
        - Pose estimation
        - YOLO for phone
    Saves processed image into different folders.
    """

    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload error: {e}")

    H, W, _ = img.shape
    reasons = {}

    phone_detected = False
    phone_conf = None
    final_valid = False

    # --------------------------------------------------------
    # 1. Detect Faces
    # --------------------------------------------------------
    bboxes = detect_faces_scrfd(img)
    num_faces = len(bboxes)
    reasons["num_faces"] = num_faces

    if num_faces == 0:
        reasons["face"] = "no_face"
        proc_img = img.copy()

        b64 = imencode_to_base64(proc_img)
        save_processed_image(proc_img, final_valid=False, phone_detected=False, prefix="invalid")

        return AnalyzeResult(
            final_valid=False,
            reasons=reasons,
            brightness=None,
            blurriness=None,
            pose=None,
            num_faces=0,
            phone_detected=False,
            phone_confidence=None,
            processed_image_base64=b64
        )

    if num_faces > 1:
        reasons["face"] = "multiple_faces"
        proc_img = img.copy()

        b64 = imencode_to_base64(proc_img)
        save_processed_image(proc_img, final_valid=False, phone_detected=False, prefix="invalid")

        return AnalyzeResult(
            final_valid=False,
            reasons=reasons,
            brightness=None,
            blurriness=None,
            pose=None,
            num_faces=num_faces,
            phone_detected=False,
            phone_confidence=None,
            processed_image_base64=b64
        )

    # --------------------------------------------------------
    # Single Face
    # --------------------------------------------------------
    x1, y1, x2, y2, score = bboxes[0]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    reasons["face_bbox"] = [x1, y1, x2, y2, float(score)]

    # --------------------------------------------------------
    # 2. 5% Border Check
    # --------------------------------------------------------
    margin_x5 = int(0.05 * W)
    margin_y5 = int(0.05 * H)

    reasons["5pct_margin_px"] = [margin_x5, margin_y5]

    if x1 <= margin_x5 or y1 <= margin_y5 or x2 >= (W - margin_x5) or y2 >= (H - margin_y5):
        reasons["face"] = "near_border_5pct"

        proc_img = draw_boxes_and_yolo_plot(img, (x1,y1,x2,y2), None)
        b64 = imencode_to_base64(proc_img)
        save_processed_image(proc_img, final_valid=False, phone_detected=False, prefix="invalid")

        return AnalyzeResult(
            final_valid=False,
            reasons=reasons,
            brightness=None,
            blurriness=None,
            pose=None,
            num_faces=num_faces,
            phone_detected=False,
            phone_confidence=None,
            processed_image_base64=b64
        )

    # --------------------------------------------------------
    # 3. Expanded 10% Crop (Brightness/Blur)
    # --------------------------------------------------------
    fw, fh = x2 - x1, y2 - y1
    margin_x10 = int(fw * 0.10)
    margin_y10 = int(fh * 0.10)

    ex1 = max(0, x1 - margin_x10)
    ey1 = max(0, y1 - margin_y10)
    ex2 = min(W, x2 + margin_x10)
    ey2 = min(H, y2 + margin_y10)

    reasons["expanded_bbox"] = [ex1, ey1, ex2, ey2]

    crop = img[ey1:ey2, ex1:ex2]

    if crop.size == 0:
        reasons["face"] = "expanded_crop_invalid"

        proc_img = draw_boxes_and_yolo_plot(img, (x1,y1,x2,y2), (ex1,ey1,ex2,ey2))
        b64 = imencode_to_base64(proc_img)
        save_processed_image(proc_img, final_valid=False, phone_detected=False, prefix="invalid")

        return AnalyzeResult(
            final_valid=False,
            reasons=reasons,
            brightness=None,
            blurriness=None,
            pose=None,
            num_faces=num_faces,
            phone_detected=False,
            phone_confidence=None,
            processed_image_base64=b64
        )

    # brightness
    brightness = compute_brightness(crop)
    reasons["brightness"] = brightness

    if not (BRIGHT_LOW <= brightness <= BRIGHT_HIGH):
        reasons["face"] = "bad_lighting"
        proc_img = draw_boxes_and_yolo_plot(img, (x1,y1,x2,y2), (ex1,ey1,ex2,ey2))

        b64 = imencode_to_base64(proc_img)
        save_processed_image(proc_img, final_valid=False, phone_detected=False, prefix="invalid")

        return AnalyzeResult(
            final_valid=False,
            reasons=reasons,
            brightness=brightness,
            blurriness=None,
            pose=None,
            num_faces=num_faces,
            phone_detected=False,
            phone_confidence=None,
            processed_image_base64=b64
        )

    # blurriness
    blurriness = compute_blurriness(crop)
    reasons["blurriness"] = blurriness

    if blurriness < BLUR_THRESHOLD:
        reasons["face"] = "blurry"
        proc_img = draw_boxes_and_yolo_plot(img, (x1,y1,x2,y2), (ex1,ey1,ex2,ey2))

        b64 = imencode_to_base64(proc_img)
        save_processed_image(proc_img, final_valid=False, phone_detected=False, prefix="invalid")

        return AnalyzeResult(
            final_valid=False,
            reasons=reasons,
            brightness=brightness,
            blurriness=blurriness,
            pose=None,
            num_faces=num_faces,
            phone_detected=False,
            phone_confidence=None,
            processed_image_base64=b64
        )

    # --------------------------------------------------------
    # 4. Pose Validation (Full Image)
    # --------------------------------------------------------
    (pitch, yaw, roll), frontal = pose_from_full_image(img)

    reasons["pose"] = {"pitch": pitch, "yaw": yaw, "roll": roll, "frontal": frontal}

    if not frontal:
        failed = []
        if abs(yaw) > YAW_THRESH: failed.append("yaw_exceeded")
        if abs(pitch) > PITCH_THRESH: failed.append("pitch_exceeded")
        if abs(roll) > ROLL_THRESH: failed.append("roll_exceeded")

        reasons["face"] = "bad_pose:" + ",".join(failed)

        proc_img = draw_boxes_and_yolo_plot(img, (x1,y1,x2,y2), (ex1,ey1,ex2,ey2))
        b64 = imencode_to_base64(proc_img)
        save_processed_image(proc_img, final_valid=False, phone_detected=False, prefix="invalid")

        return AnalyzeResult(
            final_valid=False,
            reasons=reasons,
            brightness=brightness,
            blurriness=blurriness,
            pose=(pitch, yaw, roll),
            num_faces=num_faces,
            phone_detected=False,
            phone_confidence=None,
            processed_image_base64=b64
        )

    # --------------------------------------------------------
    # 5. YOLO Phone Detection
    # --------------------------------------------------------
    results = yolo(img, verbose=False)

    phone_conf = 0.0
    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = yolo.names[cls].lower()
        if "phone" in label:
            phone_conf = max(phone_conf, conf)
            phone_detected = True

    reasons["phone_detected"] = phone_detected
    reasons["phone_confidence"] = phone_conf

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------
    final_valid = (not phone_detected)


    # --------------------------------------------------------
    # 6. SPOOF CHECK (ONLY IF PHONE NOT DETECTED)
    # --------------------------------------------------------
    spoof_status = "REAL"
    spoof_conf = 0.0

    if not phone_detected:
        # send cropped face instead of full image
        _, crop_buffer = cv2.imencode(".jpg", crop)
        crop_bytes = crop_buffer.tobytes()

        spoof_status, spoof_conf = run_neuroverify_api(crop_bytes)

        # if API failed → default to REAL
        if spoof_status == "ERROR":
            spoof_status = "REAL"
            spoof_conf = 0.0

        if spoof_status == "SPOOF":
            final_valid = False
            reasons["liveness"] = "spoof_detected"
        else:
            final_valid = True and (not phone_detected)

    else:
        # phone detected → no spoof check
        spoof_status = "SKIPPED"

    proc_img = draw_boxes_and_yolo_plot(img, (x1,y1,x2,y2), (ex1,ey1,ex2,ey2), results)
    b64 = imencode_to_base64(proc_img)

    save_processed_image(proc_img, final_valid=final_valid, phone_detected=phone_detected, prefix="processed")
    if phone_detected:
        clean_spoof = "NONE"
    else:
        if spoof_status == "SPOOF":
            clean_spoof = "SPOOF"
        elif spoof_status == "REAL":
            clean_spoof = "REAL"
        else:
            clean_spoof = "NONE"   # default fallback

    # --------------------------------------------------------
    # RETURN FINAL RESULT
    # --------------------------------------------------------
    return AnalyzeResult(
        final_valid=final_valid,
        reasons=reasons,
        brightness=brightness,
        blurriness=blurriness,
        pose=(pitch, yaw, roll),
        num_faces=num_faces,
        phone_detected=phone_detected,
        phone_confidence=phone_conf if phone_detected else None,
        processed_image_base64=b64,
        spoof_status=clean_spoof,
        spoof_confidence=None
    )


# ------------------------------------------------------------
# Health Route
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
