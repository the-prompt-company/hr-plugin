# HR Attrition Skills · v0.5.0

과거 스냅샷으로 학습·후보 비교·미래 평가를 수행하고, 고정 모델로 현재 직원의 퇴사위험 순위를 계산합니다. 직원 목록 중심 HTML에서 유지 우선순위와 HR 확인 기록을 별도로 관리합니다.

## 설치

```bash
python -m pip install -e '.[dev]'
```

Python 3.11 이상. 진단에는 pandas, numpy, openpyxl, scipy가 필요하고 예측에는 scikit-learn이 필요합니다. 의존성 오류를 진단 모드로 숨기지 않습니다.

## 1. 과거 학습 · 후보 비교 · 미래 평가

[학습 manifest 계약](skill/predict_attrition/reference/training.md)을 따라 history.json과 과거 스냅샷을 준비합니다. 각 스냅샷의 라벨 시트에는 `employee_id, left_date, left_type, observed_until`이 필요합니다. left(0/1)만 있는 파일은 학습하지 않습니다.

```bash
hrattrition train --manifest history.json --out runs/model-v1
```

- 라벨은 요청한 퇴사 유형·기간으로 생성합니다. 미성숙 관측, 자발적 분석의 경쟁 사건, 유형 불명은 제외합니다.
- 라벨 기간이 다음 분할 경계를 넘는 행을 제거합니다.
- 기준선 + 로지스틱 회귀(core/all) + HistGradientBoosting(core/all), 총 5개 후보를 검증 기간의 평균 Precision@K로 비교합니다.
- 선택 뒤 미래 평가 기간으로 Precision/Recall/Lift@K, ROC-AUC, AP, Brier, 보정 구간을 기록합니다. 미래 평가로 후보를 고르지 않습니다.
- 내보내는 모델은 평가한 모델 그대로입니다. 평가 후 재학습하지 않습니다.

산출물: `model.joblib`, `evaluation.json`, `candidate_comparison.csv`, `future_predictions.csv`.

## 2. 현재 재직자 예측

```bash
hrattrition run --input current.xlsx --model-artifact runs/model-v1/model.joblib \
  --target '자발적 퇴사' --horizon-days 90 --out runs/current --share
```

모델의 회사·유형·기간을 검사하고 현재 기준일이 모델 평가의 관측 종료일보다 늦어야 합니다. 현재 파일의 라벨은 학습에 쓰지 않습니다. 신뢰할 수 있는 모델 파일만 로드합니다.

점수는 **순위용이며 보정된 퇴사 확률이 아닙니다**. 높은 점수 순 상위 K명, 다음 K명, 나머지로 구간을 나눕니다. 직원 순위의 동률은 사번으로 일관되게 처리합니다. 유지 우선순위는 HR이 별도 지정합니다.

모델 없이 현재 신호만 진단하려면 명시적으로 실행합니다.

```bash
hrattrition run --input current.xlsx --mode heuristic --out runs/diagnostic
```

라벨이 있는 파일을 자동 재학습하지 않습니다. 예측 실패를 진단 성공으로 바꾸지 않습니다.

## 3. 대시보드

```bash
hrattrition dashboard --result runs/current/퇴사위험_결과.xlsx \
  --data current.xlsx --out runs/current --share
```

- 차콜 블랙 테마, 직원 목록 우선: 실제 값·단위·기간·비교 기준.
- 조직별 HR 지표 히트맵, 위험 구간 분포, 위험×유지 우선순위 교차표, 월별 연장근로 추이, 최근 설문 차트.
- 히트맵 색은 지표별 전체 명부 분포 백분위이며 퇴사위험 기여도가 아닙니다. 셀 클릭과 CSV 내보내기는 공통 필터를 따릅니다.
- 조직·검색·위험·유지 우선순위·상태 공통 필터: 모든 업무 패널에 동일 적용.
- 담당자·확인 상태·다음 확인일·유지 우선순위: 회사별 브라우저 저장 + JSON 내보내기/가져오기. 충돌은 기존 값을 보존합니다.
- H1~H9, V1~V9, 모델 성능: 기본 접힘. 검증은 전체 분석에 대한 고정 기록임을 명시합니다.
- 공유용: 별도 HTML에서 이름·사번을 가명으로 변경합니다. 공유용 가명은 기준일 간 연결용이 아닙니다.
- 외부 CDN 없이 동작합니다. 원본이 없는 패널은 데이터 부족 이유를 표시합니다.

## Claude 스킬

```bash
python make_skill_zip.py --check
```

`dist/predict_attrition.zip`, `dist/make_dashboard.zip`을 설치합니다. 두 ZIP은 공용 엔진을 각각 포함합니다.

스킬에서 과거 학습:

```bash
python skill/predict_attrition/scripts/predict_attrition.py --train-manifest history.json --out runs/model-v1
```

[예측 스킬](skill/predict_attrition/SKILL.md) · [대시보드 스킬](skill/make_dashboard/SKILL.md).

## 합성 데모

```bash
python tools/make_temporal_fixture.py --out artifacts/temporal-demo
hrattrition train --manifest artifacts/temporal-demo/history.json --out artifacts/temporal-demo/model
hrattrition run --input artifacts/temporal-demo/current.xlsx \
  --model-artifact artifacts/temporal-demo/model/model.joblib --out artifacts/temporal-demo/result --share
```

샘플은 가상 데이터이며 서로 다른 시점의 처리 경로를 검증하기 위한 것입니다. 실제 회사 성능이나 직원 행동의 시간적 재현을 보장하지 않습니다. 기존 `fixtures/hr_fixture_186.xlsx`도 처리 검증 전용입니다.

## 검증

```bash
python -m pytest tests -q
python make_skill_zip.py --check
# Playwright와 Chrome이 있는 환경
PLAYWRIGHT_MODULE=/path/to/playwright node tests/browser_dashboard.cjs
```

테스트는 라벨 기간·유형·미성숙 관측, 로더 날짜 오류 보존, 미래 날짜 실제 제거, recorded_at 시점 차단, 후보 선택과 테스트 라벨의 독립성, 고정 모델 추론, 모델/질문 불일치, Excel 왕복, 배포 ZIP을 검증합니다. 브라우저 테스트는 공통 필터 집계, 기록 보존·import/export, 점수 불변성, 접힘 상태, 모바일 폭을 확인합니다.

## 범위와 한계

- 실제 운영 정확도는 실제 과거 데이터와 이후 관측 결과로 검증해야 합니다. 출시 성능을 자동 승인하지 않습니다.
- 확률 보정은 자동 수행하지 않으므로 확률로 표시하지 않습니다. 평가에 Brier와 구간별 예측/관측 비율을 포함합니다.
- 여러 시점의 동일 직원과 겹치는 라벨 기간은 독립 표본이 아닙니다. 이전에 본 직원/새 직원 성능을 별도로 제공합니다.
- 스냅샷은 해당 시점의 실제 상태여야 합니다. recorded_at이 없는 원본의 소급 입력 가용성까지 시스템이 보장할 수 없습니다.
- 재입사는 고용 에피소드 ID를 일관되게 사용해야 합니다.
- 기록은 로컬 브라우저 기반이며 공동 편집 서버가 아닙니다. 회사명 중복이 가능하면 고유 company_id를 제공하세요.
- 모델 파일은 신뢰 가능한 로컬 산출물만 사용합니다.

## 변경된 구조

`labels.py` 사건 라벨 / `training.py` 시간 분할·비교·추론 / `features.py` 시점 일관성·실제 값 / `dashboard.py` HR 검토 화면 / `tools/make_temporal_fixture.py` 시점별 데모.

고도화 과거 계획은 [2026-10-19 문서](Docs/HANDOFF_HR_RETENTION_UPGRADE_2026-10-19.md)에 보존합니다. 현재 동작은 이 README와 v0.5 스킬 문서를 따릅니다.

Proprietary — The Prompt Company.
