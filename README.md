# 차체 도장·외관 불량 자동 검출 시스템

YOLO11 + OpenCV + pandas를 활용한 차체 외관 불량 검출 포트폴리오 프로젝트

## 기술 스택
- YOLO11 (ultralytics) — 불량 위치 탐지 + 분류
- OpenCV — 결과 시각화
- pandas / matplotlib — 불량률 통계 분석
- Streamlit — 데모 UI

## 데이터셋
[Roboflow — Car Dent & Scratch Detection](https://universe.roboflow.com/sindhu/car_dent_scratch_detection-1)

Roboflow에서 **YOLO 포맷**으로 다운로드 → 압축 해제하면 아래 구조로 옴:
```
data/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

## 실행 순서

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 모델 학습 (data.yaml 경로 확인 후)
python train.py

# 3. 데이터 분석
jupyter notebook analysis.ipynb

# 4. 데모 앱
streamlit run app.py
```

## 프로젝트 구조
```
car_defect_inspection/
├── src/
│   └── preprocessing.py   # OpenCV 시각화 보조
├── data/                  # Roboflow 다운로드 데이터
├── results/               # 결과 보관용
├── runs                   # 학습 결과 (best.pt, 그래프)
├── train.py               # YOLO 학습
├── analysis.ipynb         # 불량률 분석
├── app.py                 # Streamlit 데모
└── requirements.txt
```
