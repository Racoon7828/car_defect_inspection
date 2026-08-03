# 프로젝트 변경 로그

## 프로젝트 초기 설정

### 폴더 구조 생성
- `data/raw/`, `data/processed/`, `models/`, `results/`, `src/` 폴더 생성

### 초기 파일 생성 (HOG+SVM 기반)
- `src/preprocessing.py` — OpenCV 전처리, HOG 특징 추출, 윤곽선 검출
- `train.py` — SVM 학습 스크립트
- `analysis.ipynb` — pandas 불량률 분석 노트북
- `app.py` — Streamlit 데모 UI
- `requirements.txt`

---

## YOLO 전환

### 전체 코드 YOLO 기반으로 재작성
- `src/preprocessing.py` — HOG 제거, YOLO 결과 시각화용으로 변경 (`draw_results`, `resize_for_display`)
- `train.py` — sklearn SVM → ultralytics YOLO11 학습으로 전환
  - 모델: `yolo11n.pt` (nano)
  - epochs: 50, patience: 10 (조기 종료)
  - 학습 완료 후 loss/mAP 그래프 자동 저장
- `app.py` — Streamlit UI YOLO 추론 기반으로 재작성
  - 이미지 업로드 → YOLO 추론 → 바운딩박스 시각화 → 결과 표 출력
- `requirements.txt` — sklearn 제거, ultralytics 추가

---

## 데이터 정리

### Roboflow 데이터셋 경로 정리
- 다운로드 폴더: `data/Car_Dent_Scratch_Detection-1-.v9-raw_images.yolov8/`
- `train/`, `valid/`, `test/`, `data.yaml` → `data/` 바로 아래로 이동
- `data.yaml` 경로를 절대경로로 수정
- `train.py` `DATA_YAML` 경로 수정

### 데이터셋 현황 (원본)
- 클래스: 17종
- 총 이미지: 6,140장 (train 4,622 / valid 1,358 / test 160)

---

## 1차 학습

### 학습 환경
- CPU → GPU (RTX 5060) 전환
- PyTorch cu121 → cu128 재설치 (RTX 5060 Blackwell 아키텍처 호환)

### 경로 오류 수정
- ultralytics가 `runs/detect/` 자동 생성하는 구조 반영
- `train.py` `PROJECT` 경로 수정: `"results"` → `"runs/detect/results"`
- `app.py` `MODEL_PATH` 수정: 실제 `best.pt` 경로로 변경

### 1차 학습 결과
- 조기 종료: epoch 43/50
- **mAP50: 0.4427** (epoch 41)
- Precision: 0.5311 / Recall: 0.4653

---

## 클래스 불균형 분석 및 보완

### 불균형 클래스 발견
| 클래스 | 원본 수량 |
|--------|---------|
| Bodypanel-Dent | 1개 |
| Signlight-Damage | 17개 |
| pillar-dent | 23개 |

### 이미지 증강 스크립트 작성
- `augment.py` 생성
- 200개 미만 클래스 대상으로 목표 200개까지 증강
- 증강 기법: 좌우 반전, 밝기 조절, 회전(±15도), 가우시안 노이즈
- 좌우 반전 시 바운딩박스 x좌표 자동 보정

### 2차 학습 결과 (증강 후)
- 조기 종료: epoch 47/50
- **mAP50: 0.4614** (epoch 47) — +0.019 향상
- Precision: 0.5272 / Recall: 0.4745

---

## 추가 데이터셋 병합

### 외부 데이터셋 수집
| 데이터셋 | 결과 |
|---------|------|
| car-dent-detection2 | 병합 (15개 클래스 일치) |
| car-damage | 부분 병합 (31개 중 일치 클래스만 추출) |
| Damage Detection v4 | 제외 (1클래스: damage) |
| Damage Detection v5 | 제외 (2클래스: Dent, Shatter) |

### 데이터셋 병합 스크립트 작성
- `merge_datasets.py` 생성
- MD5 해시로 중복 이미지 자동 감지 및 제외
- 외부 클래스 ID → 우리 클래스 ID 자동 변환
- 클래스 불일치 항목 자동 제외

### 병합 결과
- 추가 이미지: 11,595장
- 기존 3,325장 → **총 14,920장**

### 병합 후 클래스별 현황
| 클래스 | 수량 |
|--------|------|
| Bodypanel-Dent | 203개 (1→203) |
| Front-Windscreen-Damage | 705개 |
| Headlight-Damage | 1,126개 |
| Rear-windscreen-Damage | 1,019개 |
| RunningBoard-Dent | 933개 |
| Sidemirror-Damage | 827개 |
| Signlight-Damage | 441개 (17→441) |
| Taillight-Damage | 1,057개 |
| bonnet-dent | 2,269개 |
| boot-dent | 223개 |
| doorouter-dent | 3,370개 |
| fender-dent | 2,003개 |
| front-bumper-dent | 3,797개 |
| pillar-dent | 391개 (23→391) |
| quaterpanel-dent | 1,915개 |
| rear-bumper-dent | 2,191개 |
| roof-dent | 1,073개 |

---

## 폴더 구조 정리
- `runs/detect/` 중첩 구조 제거
- 1차 학습 결과 → `runs/train_1/`
- 3차 학습 결과 → `runs/train_3/`
- `train.py` PROJECT: `"runs"`, RUN_NAME: `"train_4"` 로 수정
- `app.py` MODEL_PATH: `"runs/train_3/weights/best.pt"` 로 수정

## 3차 학습 결과 (데이터 병합 후)

- 데이터: 14,920장
- epoch: 50/50 (조기 종료 없음)
- **mAP50: 0.5883** (1차 대비 +0.146)
- Precision: 0.6466 / Recall: 0.5366
- `app.py` MODEL_PATH 3차 모델로 업데이트

---

## 코드 품질 개선 (REVIEW.md 기반)

**검토일:** 2026-06-25  
**기준 문서:** REVIEW.md (AI 정적 분석 + 기능 테스트)

### [C-1] data/data.yaml — 절대 경로 → 상대 경로

```yaml
# 변경 전
train: C:\Users\Win11Pro\Desktop\car_defect_inspection\data\train\images
val:   C:\Users\Win11Pro\Desktop\car_defect_inspection\data\valid\images
test:  C:\Users\Win11Pro\Desktop\car_defect_inspection\data\test\images

# 변경 후
train: data/train/images
val:   data/valid/images
test:  data/test/images
```

- 다른 PC에서 `python train.py` 실행 시 즉시 실패하는 이식성 문제 해결

### [C-2] augment.py — 플립 라벨 불일치 버그 수정

`augment_image()` 내부에 `choice==0` (좌우 반전)이 있었으나, 호출부(`main()`)에서 `aug_lines = lines`를 그대로 사용해 **이미지만 반전되고 라벨 x좌표가 미보정**되는 학습 데이터 오염 버그.

- `augment_image()` 에서 flip 케이스 제거 → 밝기/회전/노이즈 3가지만 유지 (`choice 0~2`)
- flip 처리는 `main()` 에서 `flip_labels_horizontal()` 와 함께 이미 올바르게 구현되어 있으므로 유지

```python
# 변경 전: choice 0~3 (0=flip 포함)
choice = random.randint(0, 3)
if choice == 0:
    img = cv2.flip(img, 1)  # 라벨 보정 없음 → 버그

# 변경 후: choice 0~2 (flip 제거)
choice = random.randint(0, 2)
# 0: 밝기, 1: 회전, 2: 가우시안 노이즈
```

### [M-1] app.py — cv2.imdecode None 방어 코드 추가

파손된 파일 업로드 시 `img_bgr = None` → `model.predict()` 에서 `AttributeError` 크래시 방지

```python
if img_bgr is None:
    st.error("이미지를 읽을 수 없습니다. 유효한 JPG/PNG 파일을 업로드하세요.")
    st.stop()
```

### [M-2] app.py — deprecated Streamlit API 교체

```python
# 변경 전 (deprecated)
st.image(..., use_column_width=True)

# 변경 후
st.image(..., use_container_width=True)
```

### [M-3] app.py — import pandas 최상단 이동

`import pandas as pd` 를 함수 블록 내부(56번째 줄)에서 파일 최상단으로 이동

### [M-4] train.py — seed=42 추가

`model.train()` 에 `seed=42` 파라미터 추가 → 재현 가능한 학습 결과 보장

### [N-1] train.py — RUN_NAME 타임스탬프 자동화

```python
# 변경 전 (매 학습마다 수동 수정 필요)
RUN_NAME = "train_4"

# 변경 후 (자동)
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"
```
