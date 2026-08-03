# 차체 도장·외관 불량 자동 검출 시스템

YOLO11 + OpenCV + pandas를 활용한 차체 외관 불량 검출 포트폴리오 프로젝트

## 기술 스택
- YOLO11 (ultralytics) — 불량 위치 탐지 + 분류
- OpenCV — 결과 시각화
- pandas / matplotlib — 불량률 통계 분석
- Streamlit — 데모 UI

## 데이터셋

원본은 [Roboflow — Car Dent & Scratch Detection](https://universe.roboflow.com/sindhu/car_dent_scratch_detection-1)이지만,
`merge_datasets.py`로 외부 데이터셋 2종을 추가 병합(6,140장 → 14,920장)하고,
`resplit_dataset.py`로 증강 이미지 유출을 막는 방식의 층화 재분할(train 80% / valid 10% / test 10%)까지 거친 상태입니다.

**`data/` 폴더는 용량 문제로 GitHub에 올라가 있지 않습니다(`.gitignore` 처리됨).**
아래 Google Drive 링크에서 `data.zip`을 받아 압축 해제 후, 프로젝트 루트에 생기는 `data/` 폴더를 그대로 사용하세요.

> Google Drive 링크: `<여기에 공유 링크 채워넣기>`

압축을 풀면 아래 구조가 됩니다:
```
data/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

새로운 데이터셋을 추가 병합하려면 [CHANGELOG.md](CHANGELOG.md)와 `merge_datasets.py` 상단의 `DATASET_MAPS`를 참고하세요.

## 실행 순서

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데모만 볼 경우 — data/ 없이 바로 실행 가능 (runs/train_3/weights/best.pt 사용)
streamlit run app.py

# 3. 학습/분석을 하려면 먼저 data.zip을 받아 data/ 에 풀어넣은 뒤
python train.py
jupyter notebook analysis.ipynb
```

## 프로젝트 구조
```
car_defect_inspection/
├── src/
│   └── preprocessing.py   # OpenCV 시각화 보조
├── data/                  # (미포함, 별도 다운로드) 병합·재분할된 학습 데이터
├── results/                # 결과 보관용
├── runs/                   # 학습 결과 (best.pt, 그래프) — runs/train_3/weights/best.pt만 git에 포함
├── train.py                # YOLO 학습
├── augment.py               # 클래스 불균형 보완용 이미지 증강
├── merge_datasets.py        # 외부 데이터셋 병합
├── resplit_dataset.py       # train/valid/test 층화 재분할
├── analysis.ipynb           # 불량률 분석
├── app.py                   # Streamlit 데모
├── REVIEW.md                 # 코드 리뷰 및 수정 이력
├── CHANGELOG.md               # 개발 변경 이력
└── requirements.txt
```
