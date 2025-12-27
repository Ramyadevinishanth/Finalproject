import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Fruit Detection", layout="centered")

st.title("🍎🍌🍊 Fruit Detection (YOLOv8)")
st.write("Upload an image to detect fruits")

# Load model 
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # place best.pt in same folder

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image using PIL
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    # Convert PIL → OpenCV format
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Run inference (MATCH COLAB)
    results = model(
        img,
        conf=0.2,
        iou=0.5,
        imgsz=640
    )

    res = results[0]

    # Show annotated image
    annotated = res.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.subheader("🔍 Detection Result")
    st.image(annotated, use_column_width=True)

    # Show detection details
    st.subheader("📊 Detection Info")
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = res.names[cls_id]
            st.write(f"**{label}** — Confidence: {conf:.2f}")
    else:
        st.write("No objects detected")
