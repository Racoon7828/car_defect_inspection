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

상세 내용은 [MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) 참고.

### README.md 갱신

데이터셋 병합·재분할 과정 반영, `data/` 폴더가 GitHub에 없다는 점과 Google Drive 다운로드 안내 추가, 프로젝트 구조에 신규 파일 반영

---

## YOLO26n v2 재학습 — 유출 없는 최종 검증 (2026-08-03)

옛 분할(`data/train`) 기준 학습된 모델로 `test_v2`를 검증했더니 두 모델 다 학습 데이터와 test_v2가 겹쳐 결과가 오염됨을 발견(자세한 내용은 [MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)). `train_v2`(12,843장, family 단위 유출 방지 처리됨)로 YOLO26n을 처음부터 50 epoch 재학습.

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

---

## 2단계 손상 종류 분류기 추가 (2026-08-04)

기존 YOLO 모델은 "부위+상태"만 구분하고 손상 종류(찌그러짐/스크래치/균열 등)를 구분하지 못하는 한계 발견(Signlight-Damage 클래스 오분석 논의 중 확인). 차량 파손 진단+수리비 상담 챗봇 계획을 위해 부위 탐지(1단계, 기존 YOLO) + 손상 종류 분류(2단계, 신규) 구조로 확장.

### 데이터: CarDD (Roboflow `car-damage-ymlgz/car-dd-coco` v8)

- crack/dent/glass shatter/lamp broken/scratch/tire flat 6종, 9,474장(train 9,021 + test 453)
- 다운로드 중 반복적인 네트워크 연결 끊김(WinError 10053/10054) 발생 — Range 헤더 기반 이어받기 스크립트로 43회 재시도 후 완전한 파일 확보, 이미지 전수 무결성 검증 통과(손상 0건)
- 라벨이 bbox/폴리곤 혼재 포맷임을 확인 후 양쪽 다 처리하도록 파싱 로직 작성

### 신규 스크립트

- `build_damage_type_crops.py` — CarDD 라벨에서 crop 생성, 이미지 단위 층화 재분할(train에서 val 10% 분리)
- `train_damage_type.py` — ResNet18 전이학습, 클래스 불균형 가중치 적용, 20epoch
- `eval_damage_type.py` — test셋 최종 평가(precision/recall/f1/confusion matrix)

### 결과

- valid 최고 정확도 0.8654(17epoch), **test 최종 정확도 0.8345**(812/973)
- glass shatter/lamp broken/tire flat은 우수, crack/dent/scratch 간 혼동이 주요 오차 원인(세 유형 모두 표면 손상이라 시각적으로 애매한 경우 존재)
- 상세 내용은 [DAMAGE_TYPE_CLASSIFIER.md](docs/DAMAGE_TYPE_CLASSIFIER.md) 참고

### app.py 통합

- YOLO 탐지 결과 각 박스를 15% 여백 crop → 분류기로 손상 종류 예측 → 결과 표에 "손상 종류" 열 추가(한글 표시)
- 분류기 체크포인트 없어도 앱이 정상 동작하도록 방어 처리
- 15장 샘플로 1단계+2단계 통합 스모크 테스트 완료(에러 0건)

### 발견된 버그: app.py 상대경로가 실행 위치(cwd)에 의존 — 수정 완료

Streamlit 서버를 프로젝트 폴더가 아닌 다른 작업 디렉토리에서 띄우면 `MODEL_PATH`/`DAMAGE_TYPE_MODEL_PATH`(상대경로)를 못 찾아, 배포된 학습 모델 대신 COCO 사전학습 기본 모델을 새로 받아버리는 문제 발견(실제 브라우저 테스트 중 발견 — 사이드바에 모델 경로가 안 뜨고 서버 로그에 `yolo11n.pt` 다운로드 로그가 찍힘).

> `BASE_DIR = Path(__file__).resolve().parent`를 도입해 두 경로 모두 스크립트 파일 기준 절대경로로 변경. 이제 `streamlit run app.py`를 어느 위치에서 실행해도 항상 올바른 모델을 찾음.

브라우저로 실제 업로드 테스트 완료: 파일 주입(DataTransfer API) 방식으로 이미지 업로드 → "불량 1건 검출" 정상 표시, 사이드바에 정확한 모델 경로와 "손상 종류 분류기: 활성화" 메시지 확인. 캔버스 기반 결과 표(glide-data-grid)라 셀 값까지 스크린샷으로 재확인은 못 했으나, 동일 이미지를 스크립트로 미리 검증(부위=Rear-windscreen-Damage, 종류=crack)해둔 결과와 탐지 건수가 일치함을 확인.

---

## 배포 모델을 YOLO11n v2로 교체 (2026-08-04)

YOLO11n vs YOLO26n 최종 비교 결과 정확도 차이가 사실상 없고(mAP50 0.827 vs 0.830) YOLO11n이 학습 속도 약 1.6배 빠른 것으로 확인되어, 배포 모델을 옛 v1 기반 `runs/train_3`(YOLO11n, mAP50 0.588)에서 **`runs/detect/runs/train_20260804_0217`(YOLO11n, v2 재학습, test mAP50 0.827)**로 교체.

- `app.py`의 `MODEL_PATH` 변경
- `.gitignore`에 새 배포 모델 경로 예외 규칙 추가 (5.5MB, git 포함)
- 로드 검증 완료(클래스 수 17개 정상 확인)
- `runs/train_3`는 삭제하지 않고 과거 기록으로 보존(REVIEW.md/CHANGELOG 초반 히스토리와 연결됨)

### 기타

- `requirements.txt`에 `torch`, `torchvision` 명시 추가
- `.gitignore`에 `cardd_raw/`, `damage_type_crops/`(대용량 데이터) 추가. `runs/damage_type_classifier/best.pt`(44.8MB)는 데모 실행에 필요해 포함(포함 규칙 누락되어 있던 것 재수정)
- 다운로드 임시 파일(`cardd_raw.zip` 1.3GB, 트러블슈팅 스크립트/로그) 정리

---

## YOLO11n v2 재학습 — 완전한 공정 비교 완성 (2026-08-04)

MODEL_COMPARISON.md에 미완으로 남아있던 "YOLO11n의 v2 기준 재학습"을 진행(`train.py`, run: `train_20260804_0217`, 동일 v2 데이터·50epoch).

- **test 최종 비교**: YOLO11n mAP50 0.827 vs YOLO26n mAP50 0.830 — **사실상 동률**
- 이전에 epoch21 시점 부분 비교에서 "YOLO26n이 Recall 크게 우세"라고 봤던 결론을 정정 — 50epoch 완주 시 격차가 사실상 사라짐(초반 수렴 속도 차이였을 뿐)
- boot-dent 클래스만 YOLO26n이 확실히 우세(mAP50 0.262→0.318), 나머지는 근소한 차이
- 결론: 정확도만으론 모델 선택 근거가 약함 — 학습 속도(YOLO11n 유리)와 배포 환경(CPU 배포 시 YOLO26n 유리 가능성, 미검증)을 기준으로 판단 권장
- 상세 내용은 [MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) 최종 섹션 참고

---

## 배포 모델을 YOLO11n v2로 교체, 결과 표시 정리 (2026-08-04)

YOLO11n과 YOLO26n의 v2 기준 정확도가 사실상 동률로 확인됨에 따라, 학습 속도가 약 1.6배 빠른 YOLO11n(v2 재학습본, run: `train_20260804_0217`, test mAP50 0.827)을 배포 모델로 확정.

- `app.py`의 `MODEL_PATH`를 `runs/train_3`(v1 기반, mAP50 0.588)에서 새 v2 학습본으로 변경
- `.gitignore`에 새 배포 모델 경로 예외 규칙 추가, 로드 검증 완료
- `runs/train_3`는 삭제하지 않고 과거 기록으로 보존
- "부위" 열에서 손상 상태 표현(찌그러짐/파손 등) 제거 — 이제 2단계 분류기가 "손상 종류" 열로 별도 표시하므로 중복이라 판단. `KOREAN_NAMES`를 부위명만 남기도록 정리 (예: "전방 범퍼 찌그러짐" → "전방 범퍼")

## 프로젝트 파일 구조 정리

파일이 늘어나 루트가 복잡해져 스크립트와 문서를 하위 폴더로 재구성.

- `scripts/` 신설 — `train.py`, `augment.py`, `merge_datasets.py`, `resplit_dataset.py`, `build_damage_type_crops.py`, `train_damage_type.py`, `eval_damage_type.py` 이동
- `docs/` 신설 — `REVIEW.md`, `MODEL_COMPARISON.md`, `DAMAGE_TYPE_CLASSIFIER.md`, `DEFECT_CLASSES.md` 이동
- `app.py`, `README.md`, `requirements.txt`는 루트 유지
- 이미 git에 추적되던 파일은 `git mv`로 이동해 히스토리 보존
- README.md 실행 명령어(`python scripts/train.py` 등), 프로젝트 구조도, 문서 간 상호 링크(`docs/` 경로 반영) 전부 갱신
- 스크립트 내부 경로(`data/`, `runs/` 등)는 CWD 기준이라 위치 이동과 무관하게 정상 동작 확인(항상 프로젝트 루트에서 실행하는 기존 관례 유지 시 문제없음)

### CHANGELOG.md, Claude.md 도 docs/로 이동

`docs/`로 마저 옮김에 따라 참조 경로 추가 수정:
- README.md의 `CHANGELOG.md` 링크·구조도를 `docs/CHANGELOG.md` 기준으로 갱신
- `.gitignore`의 `Claude.md` 제외 규칙은 슬래시 없는 패턴이라 `docs/Claude.md`도 그대로 자동 적용됨을 `git check-ignore`로 확인 — 별도 수정 불필요
- 다른 문서(MODEL_COMPARISON.md, REVIEW.md)의 `CHANGELOG.md` 언급은 실제 마크다운 링크가 아니라 본문 텍스트라 경로 수정 불필요

---

## README.md — 스크립트별 용도 및 재학습 방법 문서화

1단계(부위 탐지)와 2단계(손상 종류 분류)가 완전히 독립된 파이프라인이라, 처음부터 재학습하려면 `scripts/train.py`와 `scripts/train_damage_type.py`를 각각 실행해야 함을 명확히 문서화.

- README.md에 "스크립트별 용도 및 재학습 방법" 섹션 신설 — 1단계/2단계 스크립트를 표로 정리하고 각각 언제 쓰는지, 필수/선택 여부 명시
- "처음부터 전부 재학습하려면" 예시 명령어 추가 (`train.py`만 돌리면 손상 종류 분류기는 기존 것이 그대로 쓰인다는 점 강조)
- `damage_type_crops.zip`(2단계 학습 데이터) Google Drive 다운로드 안내를 데이터셋 섹션에 추가
- "실행 순서" 섹션도 `train_damage_type.py` 누락되어 있던 것 반영

---

## README.md — 모델 성능 요약 표 추가

프로젝트에 존재하는 `best.pt` 체크포인트(YOLO11n v2, YOLO26n v2, ResNet18)의 성능 수치와 git 포함 여부를 한눈에 볼 수 있는 표를 README.md에 추가.

- YOLO11n(v1, 구버전 `runs/train_3`)은 유출 있는 옛 분할 기준이라 다른 모델과 나란히 비교하면 오해 소지가 있어 표에서 제외(과거 기록은 REVIEW.md/CHANGELOG에 그대로 남아있음)
- 각 모델의 상세 근거는 MODEL_COMPARISON.md/DAMAGE_TYPE_CLASSIFIER.md로 링크

### 지표 세분화 및 설명 추가

- mAP50만 있던 표에 mAP50-95, Precision, Recall 열 추가 (1단계), 분류기(2단계)는 정확도 + macro Precision/Recall/F1로 별도 표 구성
- ResNet18의 macro Precision/Recall/F1은 DAMAGE_TYPE_CLASSIFIER.md의 클래스별 수치(6개 클래스)를 단순 평균해 계산: P 0.838, R 0.851, F1 0.843 — 이전 표에 잘못 기입되어 있던 YOLO26n 수치(0.835/0.796)를 올바른 값으로 정정
- "지표 설명" 섹션 신설 — mAP50/mAP50-95/Precision/Recall(탐지)과 정확도/macro P·R·F1(분류)의 의미를 각각 설명
