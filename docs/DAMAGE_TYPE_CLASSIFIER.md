# 2단계 손상 종류 분류기 (2026-08-04)

## 배경

기존 YOLO 모델(17개 클래스)은 "부위+상태"만 구분하고 손상 "종류"(찌그러짐/스크래치/균열 등)는 구분하지 못했음(DEFECT_CLASSES.md에서 확인). 향후 계획 중인 "AI 차량 파손 진단 + 수리비 상담 챗봇"에서 정확한 견적을 내려면 부위뿐 아니라 손상 종류까지 알아야 하므로, 별도의 2단계 분류기를 추가함.

## 전체 구조

```
이미지 업로드
  → [1단계] YOLO26n(v2 학습본) — "어느 부위"에 손상이 있는지 탐지
  → 탐지된 박스를 여백(15%) 포함해 crop
  → [2단계] ResNet18 분류기 — 그 crop이 "어떤 종류"의 손상인지 분류
  → 최종 표시: "부위 + 종류" (예: "전방 범퍼 - 스크래치")
```

## 데이터: CarDD (Roboflow `car-damage-ymlgz/car-dd-coco`, version 8)

- 손상 "종류" 기준으로 라벨링된 공개 데이터셋(CC BY 4.0), 6개 클래스: crack, dent, glass shatter, lamp broken, scratch, tire flat
- train 9,021장 / test 453장 (valid split 없음 — 직접 분리)
- 라벨은 bbox와 폴리곤(세그멘테이션)이 클래스별로 혼재된 포맷 — `build_damage_type_crops.py`가 두 포맷 모두 axis-aligned bbox로 변환해 처리
- 다운로드 중 Windows 네트워크 연결이 반복적으로 끊기는 문제 발생(`ConnectionResetError` 10053/10054) → Range 헤더 기반 이어받기 스크립트로 43회 재시도 끝에 완전한 파일 확보, 이미지 9,474장 전수 무결성 검증(PIL verify) 통과

## 파이프라인 스크립트

| 스크립트 | 역할 |
|---|---|
| `build_damage_type_crops.py` | CarDD 원본에서 이미지 단위 층화 재분할(train 내 val 10% 분리, family/유출 방지 원리는 `resplit_dataset.py`와 동일) 후, 라벨 영역을 15% 여백 포함해 crop, `damage_type_crops/{train,val,test}/<class>/*.jpg` 형태로 저장 |
| `train_damage_type.py` | ResNet18(ImageNet 사전학습) 전이학습, 클래스 불균형 보정 가중치 적용, 20epoch(patience=5) |
| `eval_damage_type.py` | test셋 최종 평가, 클래스별 precision/recall/f1 + confusion matrix 출력 |

## 데이터 규모 (crop 생성 결과)

| 클래스 | train | val | test |
|---|---|---|---|
| dent | 7,311 | 772 | 226 |
| scratch | 6,373 | 755 | 462 |
| crack | 2,565 | 277 | 88 |
| lamp broken | 1,109 | 124 | 83 |
| glass shatter | 999 | 111 | 105 |
| tire flat | 585 | 63 | 9 |

## 학습 결과

- 20epoch 완주(조기종료 미발동), 최고 val acc **0.8654** (17epoch)
- 학습 곡선: `runs/damage_type_classifier/training_curve.png`

## test셋 최종 평가 (신뢰 가능, val과 별개의 held-out set)

**종합 정확도: 812/973 = 0.8345**

| 클래스 | Precision | Recall | F1 | 개수 |
|---|---|---|---|---|
| glass shatter | 0.990 | 0.905 | 0.945 | 105 |
| lamp broken | 0.832 | 0.952 | 0.888 | 83 |
| tire flat | 0.889 | 0.889 | 0.889 | 9 |
| scratch | 0.862 | 0.848 | 0.855 | 462 |
| dent | 0.782 | 0.761 | 0.771 | 226 |
| crack | 0.673 | 0.750 | 0.710 | 88 |

### Confusion Matrix 분석

- **glass shatter/lamp broken/tire flat은 매우 정확함** — 시각적으로 뚜렷이 구별되는 손상이라 예상대로 잘 분류됨
- **crack/dent/scratch 간 혼동이 주요 오차 원인**: dent→scratch 45건, scratch→dent 39건, scratch→crack 21건, crack→scratch 14건. 세 유형 모두 "표면 손상"이라 각도·조명에 따라 시각적으로 애매한 경우가 실제로 존재함 — 데이터 품질보다는 태스크 자체의 본질적 난이도로 보임
- crack의 Precision(0.673)이 가장 낮음 — 다른 유형을 crack으로 오분류하는 경우는 적지만(위 confusion matrix 참고), scratch를 crack으로 잘못 예측하는 경우가 상대적으로 많음(21건)

## app.py 통합

- `runs/damage_type_classifier/best.pt`를 `@st.cache_resource`로 로드
- YOLO 탐지 박스마다 15% 여백 crop → 분류기 추론 → 한글 라벨(`DAMAGE_TYPE_KOREAN`)로 변환해 결과 표에 "손상 종류" 열로 추가
- 분류기 체크포인트가 없어도 앱이 깨지지 않도록 방어 처리(`load_damage_type_model()`이 `(None, None)` 반환 시 종류 열 생략)
- 15장 샘플로 1단계+2단계 통합 파이프라인 스모크 테스트 완료(에러 0건)

## 알려진 한계 및 향후 개선

1. **crack/dent/scratch 혼동** — 세 클래스 간 정확도가 상대적으로 낮음(F1 0.71~0.86). 개선하려면 손상 경계 부분 확대 crop, 또는 세 클래스만 별도로 더 정교한 특징을 학습하는 보조 모델 고려 가능
2. **tire flat 클래스의 프로젝트 적합성 검토 필요** — 차체 외관(범퍼/도어/유리 등) 진단이 목적이라면 타이어 펑크는 범위 밖일 수 있음. 필요시 `DAMAGE_TYPE_KOREAN`에서 제외하거나 무시하도록 후처리 가능
3. **YOLO 탐지 박스 여백(padding) 민감도 미검증** — 15%는 학습 시 사용한 값을 그대로 재사용했지만, 실제 YOLO가 그리는 박스 크기/여백과 정확히 일치하지 않을 수 있어 실제 이미지로 추가 검증 권장
