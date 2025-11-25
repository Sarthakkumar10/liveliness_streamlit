# frontend/streamlit_app.py
import streamlit as st
import requests
import base64
import cv2
import numpy as np
from PIL import Image
import io

# BACKEND_URL = "http://localhost:8000/analyze_frame"
BACKEND_URL = "https://2fd7f83bcd45.ngrok-free.app/analyze_frame"

st.set_page_config(layout="wide", page_title="Face+Phone Validator")

st.title("Face + Phone Validation (Streamlit UI)")
st.write("Capture from camera or upload an image. Backend runs SCRFD + Mediapipe + YOLO.")

col1, col2 = st.columns([1, 1])
with col1:
    img_file = st.camera_input("Use your webcam", disabled=False)
    st.write("Or upload an image:")
    upload = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])

    # prefer camera input if present
    input_image = img_file if img_file is not None else upload

    if input_image is None:
        st.info("Provide an image from webcam or upload one.")
with col2:
    st.markdown("### Result")
    status_placeholder = st.empty()
    debug_placeholder = st.empty()
    img_placeholder = st.empty()

if st.button("Send to backend"):

    if input_image is None:
        st.error("No image provided.")
    else:
        # prepare file bytes
        img_bytes = input_image.getvalue()
        files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
        with st.spinner("Sending to backend..."):
            try:
                resp = requests.post(BACKEND_URL, files=files, timeout=30)
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if resp.status_code != 200:
            st.error(f"Backend error: {resp.status_code} {resp.text}")
            st.stop()

        data = resp.json()
        # show parsed details
        final_valid = data.get("final_valid", False)
        reasons = data.get("reasons", {})
        brightness = data.get("brightness")
        blurriness = data.get("blurriness")
        pose = data.get("pose")

        # status light
        if final_valid:
            status_placeholder.markdown("<h2 style='color:green'>🟢 VALID IMAGE</h2>", unsafe_allow_html=True)
        else:
            status_placeholder.markdown("<h2 style='color:red'>🔴 INVALID IMAGE</h2>", unsafe_allow_html=True)

        # human-readable debug
        debug_md = "#### Debug info\n"
        debug_md += f"- num_faces: {reasons.get('num_faces')}\n"
        debug_md += f"- face_reason: {reasons.get('face','OK')}\n"
        if brightness is not None:
            debug_md += f"- brightness: {brightness:.1f}\n"
        if blurriness is not None:
            debug_md += f"- blurriness: {blurriness:.1f}\n"
        if pose:
            p, y, r = pose
            debug_md += f"- pose (P,Y,R): {p:.1f}, {y:.1f}, {r:.1f}\n"
        debug_md += f"- phone_detected: {data.get('phone_detected')}\n"
        if data.get("phone_confidence") is not None:
            debug_md += f"- phone_confidence: {data.get('phone_confidence'):.2f}\n"
        debug_md += "\n**Raw reasons object:**\n"
        debug_md += f"```\n{reasons}\n```\n"

        debug_placeholder.markdown(debug_md)

        # display processed image if present
        pimg_b64 = data.get("processed_image_base64")
        if pimg_b64:
            pimg = base64.b64decode(pimg_b64)
            image = Image.open(io.BytesIO(pimg))
            img_placeholder.image(image, caption="Processed image (YOLO + face boxes)", use_column_width=True)
        else:
            st.info("No processed image available.")
