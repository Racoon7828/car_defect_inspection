"""
YOLO 모델 학습 스크립트
사용법: python train.py

Roboflow에서 YOLO 포맷으로 다운로드하면 data.yaml이 자동 생성됨.
data.yaml 위치를 DATA_YAML 변수에 지정 후 실행.
"""
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

DATA_YAML = "data/data.yaml"
MODEL     = "yolo11n.pt"       # nano: 가볍고 빠름 / yolo11s.pt: 조금 더 정확
EPOCHS    = 50
IMG_SIZE  = 640
PROJECT   = "runs"
RUN_NAME  = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"  # 자동 타임스탬프


def main():
    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=PROJECT,
        name=RUN_NAME,
        patience=10,       # 10 epoch 개선 없으면 조기 종료
        batch=16,
        workers=4,
        seed=42,
        exist_ok=True,
    )

    print("\n=== 학습 완료 ===")
    print(f"최적 모델 저장 위치: {PROJECT}/{RUN_NAME}/weights/best.pt")

    # 학습 결과 CSV → pandas 분석
    csv_path = f"{PROJECT}/{RUN_NAME}/results.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train/box_loss"], label="train box loss")
    axes[0].plot(df["epoch"], df["val/box_loss"],   label="val box loss")
    axes[0].set_title("Box Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50", color="green")
    axes[1].set_title("mAP@50")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{PROJECT}/{RUN_NAME}/training_curve.png", dpi=150)
    print(f"학습 곡선 저장: {PROJECT}/{RUN_NAME}/training_curve.png")


if __name__ == "__main__":
    main()
