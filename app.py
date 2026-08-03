"""
Streamlit 데모 앱 — YOLO 차체 외관 불량 검출
실행: streamlit run app.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from pathlib import Path

from src.preprocessing import draw_results, resize_for_display

MODEL_PATH = "runs/train_3/weights/best.pt"  # 3차 학습 mAP50: 0.5883
FALLBACK   = "yolo11n.pt"   # 학습 전 테스트용 기본 모델

st.set_page_config(page_title="차체 도장 불량 검출", layout="wide")
st.title("차체 도장·외관 불량 자동 검출 시스템")
st.caption("Roboflow Car Defect Dataset + YOLO11 + OpenCV")

@st.cache_resource
def load_model():
    path = MODEL_PATH if Path(MODEL_PATH).exists() else FALLBACK
    return YOLO(path), path

model, used_path = load_model()
st.sidebar.info(f"사용 모델: `{used_path}`")

conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)

uploaded = st.file_uploader("차체 이미지 업로드 (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("이미지를 읽을 수 없습니다. 유효한 JPG/PNG 파일을 업로드하세요.")
        st.stop()

    MIN_SIZE = 100  # px — 이보다 작으면 오탐 위험이 커서 차단
    h, w = img_bgr.shape[:2]
    if h < MIN_SIZE or w < MIN_SIZE:
        st.error(f"이미지가 너무 작습니다 ({w}x{h}). 최소 {MIN_SIZE}x{MIN_SIZE} 이상 업로드하세요.")
        st.stop()

    results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
    vis = draw_results(img_bgr, results)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본")
        st.image(cv2.cvtColor(resize_for_display(img_bgr), cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("검출 결과")
        st.image(cv2.cvtColor(resize_for_display(vis), cv2.COLOR_BGR2RGB), use_container_width=True)

    # 탐지 결과 요약
    boxes = results[0].boxes
    st.divider()
    if len(boxes) == 0:
        st.success("불량 미검출 — 정상 판정")
    else:
        defects = [results[0].names[int(b.cls[0])] for b in boxes]
        confs   = [float(b.conf[0]) for b in boxes]
        st.error(f"불량 {len(boxes)}건 검출")
        df = pd.DataFrame({"결함 유형": defects, "신뢰도": [f"{c:.2%}" for c in confs]})
        st.dataframe(df, use_container_width=True)
