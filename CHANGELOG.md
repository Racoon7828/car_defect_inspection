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

---

## 잔여 이슈 수정 및 데이터셋 재분할 (2026-08-03)

**검토 기준:** REVIEW.md

### [M-5] app.py — 최소 이미지 크기 검증 추가

100x100px 미만 이미지 업로드 시 차단하는 방어 코드 추가 (엣지 케이스 오탐 방지)

### [M-6] analysis.ipynb Section 4 — 실제 추론 결과로 교체

`np.random` 가상 시뮬레이션 데이터를 제거하고, test셋(80장)에 실제 YOLO 추론을 돌려 5장 단위 배치로 묶어 불량률 추이 시각화로 교체

### [N-2] merge_datasets.py — 파일명 접미사 해시화

`ds_folder[:8]` 앞 8글자 절단 방식 → `hashlib.md5(ds_folder)` 해시(8자) + `split` 조합으로 변경, 파일명 충돌 위험 제거

### data/data.yaml — 경로 재해석 버그 수정

ultralytics 버전 업그레이드로 인해 `data.yaml`(이 `data/` 폴더 안에 위치)의 상대경로가 `data/` + `data/train/images`로 중복 해석되어 학습이 실패하는 문제 발견. `train/val/test` 경로에서 중복된 `data/` 접두사 제거로 해결 (`data/train/images` → `train/images`).

### resplit_dataset.py 신설 — 데이터셋 층화 재분할

기존 train(15,210)/valid(679)/test(80) 분할은 test셋이 지나치게 작고(14 인스턴스, 6/17 클래스만 등장) 신뢰도가 낮았음.

- 전체 이미지를 "family"(원본 + `augment.py` 증강본) 단위로 그룹화해, 증강 이미지가 원본과 다른 split에 들어가는 데이터 유출을 방지
- family가 포함한 클래스 중 전역적으로 가장 희귀한 클래스를 기준으로 그룹화 후 80/10/10 분할 → 희귀 클래스도 valid/test에 비례 배정
- 결과: train_v2 12,843 / valid_v2 1,551 / test_v2 1,575장
- 발견: `Bodypanel-Dent` 클래스는 family가 4개뿐 — 기존 "203개 인스턴스"는 원본 사진 3~4장을 증강으로 부풀린 것으로 확인. 원본 데이터 자체의 다양성 부족.

### DEFECT_CLASSES.md 신설

17개 탐지 클래스를 카테고리(덴트/램프/유리/미러)별로 정리한 문서 추가

### YOLO26n 비교 실험

기존 YOLO11n(Recall 0.537로 낮음) 대비 차세대 아키텍처 YOLO26n 비교 학습 진행. 동일 조건(같은 데이터, epoch 21)에서 YOLO26n이 mAP50 +4.5%, Recall +13.4%, mAP50-95 +6.6% 우세, Precision만 소폭 하락. 21/50 epoch에서 학습 중단(성능 개선 추세 확인 후 조기 종료 판단). epoch당 학습 시간은 YOLO26n이 약 1.6배 느림.

`test_v2`로 두 모델을 재검증 시도했으나, 두 모델 모두 옛 분할(`data/train`) 기준으로 학습되어 `test_v2`와 학습 데이터가 겹치는 데이터 유출 발견 — 해당 결과는 무효 처리. `train_v2` 기준 재학습 없이는 공정한 재검증 불가.

상세 내용은 [MODEL_COMPARISON.md](MODEL_COMPARISON.md) 참고.

### README.md 갱신

데이터셋 병합·재분할 과정 반영, `data/` 폴더가 GitHub에 없다는 점과 Google Drive 다운로드 안내 추가, 프로젝트 구조에 신규 파일 반영

---

## YOLO26n v2 재학습 — 유출 없는 최종 검증 (2026-08-03)

옛 분할(`data/train`) 기준 학습된 모델로 `test_v2`를 검증했더니 두 모델 다 학습 데이터와 test_v2가 겹쳐 결과가 오염됨을 발견(자세한 내용은 [MODEL_COMPARISON.md](MODEL_COMPARISON.md)). `train_v2`(12,843장, family 단위 유출 방지 처리됨)로 YOLO26n을 처음부터 50 epoch 재학습.

- `data/data_v2.yaml` 신설 (train_v2/valid_v2/test_v2 참조)
- `train.py`의 `DATA_YAML`을 `data/data_v2.yaml`로 변경
- 재학습 전 split 정합성(이미지 수, 파일명 중복, family 유출) 전수 검증 — 문제 없음 확인
- **valid_v2 최종(50epoch)**: mAP50 0.881, Precision 0.874, Recall 0.831
- **test_v2 최종 검증(유출 없음, 신뢰 가능)**: mAP50 0.830, mAP50-95 0.707, Precision 0.835, Recall 0.796
- YOLO11n은 시간 관계상 v2 재학습 미실시 — 필요 시 추후 동일 절차로 진행 예정

### 발견된 버그: train.py 결과 경로 불일치 — 수정 완료 (2026-08-03)

학습 후처리(학습곡선 그래프 저장) 단계에서 `FileNotFoundError` 발생. `train.py`가 `results.csv` 경로를 `f"{PROJECT}/{RUN_NAME}/..."`로 수동 조합하는데, 실제 ultralytics가 저장하는 경로는 `runs/detect/runs/{RUN_NAME}/...`로 한 단계 더 중첩됨(버전 변화로 인한 구조 차이, `data.yaml` 경로 버그와 같은 계열). 모델 가중치 저장 자체는 정상이었음.

> `model.train()`의 반환값(`DetMetrics`)에는 `save_dir` 속성이 없어(`results.save_dir` 시도는 실패) `model.trainer.save_dir`을 사용하도록 수정. 경로를 수동 조합하지 않고 실제 트레이너가 사용한 경로를 그대로 참조하므로 향후 ultralytics 버전이 다시 바뀌어도 안전함.
> 1 epoch/데이터 2% 스모크 테스트로 정상 동작 확인 (`SAVE_DIR`가 `runs/detect/runs/smoketest`로 정확히 출력, CSV 로드·그래프 저장 성공).

### 프로젝트 정리

불필요해진 파일/폴더 삭제:
- `train_yolo26n.log` (중단된 옛 실험 로그, 수치는 MODEL_COMPARISON.md에 기록됨)
- `data/data_abs.yaml`, `data/data_test_v2.yaml` (트러블슈팅용 임시 yaml, `data_v2.yaml`로 대체)
- `data/processed/`, `data/raw/` (빈 폴더, 미사용 — REVIEW.md N-3에서 지적된 항목)
- `runs/detect/runs/smoketest/`, `train_20260803_1639/`(최초 실패 시도), `train_20260803_1707/`(중단된 옛 분할 21epoch 실험, v2 결과로 대체됨)
- `runs/detect/val`, `val-2`~`val-5` (데이터 유출로 무효 처리된 예전 검증 결과물)

`runs/detect/runs/train_20260803_1918/`(v2 최종 학습)과 `runs/detect/val-6/`(v2 최종 검증 결과물)는 보존.

### v2 데이터셋을 기본값으로 승격

v2는 v1(옛 train+valid+test)을 그대로 풀(pool)로 모아 재배열한 것이라 이미지 손실 없음을 확인 후 진행.

- `data/train`, `data/valid`, `data/test`(v1) 삭제
- `data/train_v2` → `train`, `valid_v2` → `valid`, `test_v2` → `test`로 이름 변경 (12,843 / 1,551 / 1,575장)
- `data/data_v2.yaml` 삭제 — `data/data.yaml`이 기존 경로(`train/images` 등) 그대로 v2 내용을 가리키므로 별도 yaml 불필요
- `train.py`의 `DATA_YAML`을 `data/data.yaml`로 원복
- `merge_datasets.py`, `augment.py`는 수정 없이도 `data/train/`을 그대로 참조하므로 자동으로 v2 기준으로 동작
- `check_det_dataset`으로 경로 해석 및 이미지 개수 재검증 완료

이제부터 `data/`는 유출 없는 층화 재분할본이 기본값임. 옛 v1 기준으로 학습된 `runs/train_3`(YOLO11n, 배포 모델)과의 재현은 더 이상 불가하지만, 관련 수치는 REVIEW.md/MODEL_COMPARISON.md/CHANGELOG.md에 이미 기록되어 있어 히스토리 손실 없음.
