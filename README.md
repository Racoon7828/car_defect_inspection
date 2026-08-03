# 차체 도장·외관 불량 자동 검출 시스템

YOLO + OpenCV + pandas를 활용한 차체 외관 불량 검출 포트폴리오 프로젝트.
1단계(부위 탐지) + 2단계(손상 종류 분류) 구조로, "어디에 + 어떤 손상"까지 알 수 있음.

## 기술 스택
- YOLO (ultralytics, 1단계 부위 탐지) — 불량 위치 탐지 + 부위 분류
- ResNet18 (torchvision, 2단계 손상 종류 분류) — 찌그러짐/스크래치/균열 등 구분 ([docs/DAMAGE_TYPE_CLASSIFIER.md](docs/DAMAGE_TYPE_CLASSIFIER.md))
- OpenCV — 결과 시각화
- pandas / matplotlib — 불량률 통계 분석
- Streamlit — 데모 UI

## 데이터셋

원본은 [Roboflow — Car Dent & Scratch Detection](https://universe.roboflow.com/sindhu/car_dent_scratch_detection-1)이지만,
`scripts/merge_datasets.py`로 외부 데이터셋 2종을 추가 병합(6,140장 → 14,920장)하고,
`scripts/resplit_dataset.py`로 증강 이미지 유출을 막는 방식의 층화 재분할(train 80% / valid 10% / test 10%)까지 거친 상태입니다.

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

새로운 데이터셋을 추가 병합하려면 [CHANGELOG.md](docs/CHANGELOG.md)와 `scripts/merge_datasets.py` 상단의 `DATASET_MAPS`를 참고하세요.

## 실행 순서

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데모만 볼 경우 — data/ 없이 바로 실행 가능 (배포 모델: YOLO11n v2, test mAP50 0.827)
streamlit run app.py

# 3. 학습/분석을 하려면 먼저 data.zip을 받아 data/ 에 풀어넣은 뒤 (항상 프로젝트 루트에서 실행)
python scripts/train.py
jupyter notebook analysis.ipynb
```

## 프로젝트 구조
```
car_defect_inspection/
├── src/
│   └── preprocessing.py         # OpenCV 시각화 보조
├── scripts/                     # 학습/데이터 처리 스크립트 (항상 프로젝트 루트에서 실행)
│   ├── train.py                    # YOLO 학습 (1단계 부위 탐지)
│   ├── augment.py                  # 클래스 불균형 보완용 이미지 증강
│   ├── merge_datasets.py           # 외부 데이터셋 병합
│   ├── resplit_dataset.py          # train/valid/test 층화 재분할
│   ├── build_damage_type_crops.py  # CarDD에서 손상 종류 분류용 crop 데이터셋 생성 (2단계)
│   ├── train_damage_type.py        # 손상 종류 분류기(ResNet18) 학습 (2단계)
│   └── eval_damage_type.py         # 손상 종류 분류기 test셋 평가 (2단계)
├── docs/                         # 문서
│   ├── REVIEW.md                    # 코드 리뷰 및 수정 이력
│   ├── MODEL_COMPARISON.md          # YOLO11n vs YOLO26n 비교
│   ├── DAMAGE_TYPE_CLASSIFIER.md    # 2단계 분류기 상세 문서
│   ├── DEFECT_CLASSES.md            # 17개 탐지 클래스 정리
│   ├── CHANGELOG.md                 # 개발 변경 이력
│   └── Claude.md                    # Claude Code 작업 규칙
├── data/                         # (미포함, 별도 다운로드) 병합·재분할된 학습 데이터
├── results/                      # 결과 보관용
├── runs/                         # 학습 결과 (best.pt, 그래프) — 배포 모델만 git에 포함
├── analysis.ipynb                # 불량률 분석
├── app.py                        # Streamlit 데모 (1+2단계 통합)
└── requirements.txt
```
