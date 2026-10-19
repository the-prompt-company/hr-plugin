# 과거 학습과 미래 평가

하나의 xlsx에 현재 직원과 퇴사 라벨을 넣어 재학습하지 않는다. 과거 기준일마다 당시 재직자와 당시 사용 가능했던 피처를 복원한 스냅샷을 준비한다. 파일 자체의 기준일을 임의로 바꾸어 과거를 만들면 안 된다.

```json
{
  "target": "자발적 퇴사",
  "horizon_days": 90,
  "top_k": 20,
  "validation_start": "2024-01-01",
  "test_start": "2025-01-01",
  "observed_through": "2025-08-01",
  "synthetic": false,
  "snapshots": [
    {"input": "snapshot-2023-01.xlsx"},
    {"input": "snapshot-2023-04.xlsx"},
    {"input": "snapshot-2024-01.xlsx"},
    {"input": "snapshot-2024-04.xlsx"},
    {"input": "snapshot-2025-01.xlsx"},
    {"input": "snapshot-2025-04.xlsx"}
  ]
}
```

경로는 manifest 위치 기준이다. 각 snapshot 항목의 `labels`로 별도 CSV를 지정할 수 있다. 명부형은 항목에 `as_of`를 지정한다. 재입사는 고용 에피소드 ID를 employee_id로 일관되게 사용한다.

- 학습: validation_start 이전이며 라벨 기간 끝도 validation_start 이전인 행.
- 후보 선택: validation_start 이상 test_start 미만이며 라벨 기간 끝이 test_start 이전인 행.
- 미래 평가: test_start 이상이고 라벨 기간 끝이 observed_through 이내인 행.
- 경계에 걸친 미성숙 행은 제외 건수를 기록한다. 관측 미완료·유형 불명·경쟁 사건도 제외한다.
- 동일 직원의 여러 시점은 허용한다. 직원 ID는 피처가 아니다. 이전에 본 직원과 새 직원 성능을 나누며, 반복 관측·겹친 예측기간을 독립 표본으로 해석하지 않는다.

후보는 빈도 기준선, 로지스틱 회귀(core/all), HistGradientBoosting(core/all)의 5개다. 전처리는 학습 구간에서만 fit한다. 검증 기간 평균 Precision@K를 우선하고 AP·Brier를 동률 기준으로 사용한다. 최종 미래 평가 결과로 모델을 다시 선택하지 않는다. 검증 구간에서 피처를 섞었을 때 Precision@K 변화도 기록한다(인과 효과 아님).

내보내는 모델은 평가한 그대로이며, 미래 평가 뒤 재학습하지 않는다. 현재 예측 기준일은 observed_through보다 늦어야 한다. 모델 파일, 컬럼 목록, 학습 중앙값, 질문 정의, 원본 해시, 분할 기간, 모델 ID를 함께 저장한다.

산출물:
- `evaluation.json`: 기간별 Precision/Recall/Lift@K, ROC-AUC, AP, Brier, 보정 구간, 기준선 비교, 직원 그룹별 평가. pooled는 관측 행 기준이며 기간별 성능을 대체하지 않는다.
- `candidate_comparison.csv`: 검증 기간 후보·피처 집합 비교.
- `future_predictions.csv`: 학습에 사용하지 않은 미래 구간의 직원·기준일·정답·예측 점수.
- `model.joblib`: 고정 모델. 신뢰할 수 있는 출처의 파일만 로드한다.

자동 확률 보정은 하지 않는다. 확률로 표시하기 전 별도 보정/평가 구간이 필요하다. 현재 제품은 점수와 순위로 표시한다. 시간 평가를 했다는 사실은 출시 성능 보장이 아니다. 표본 수·기간별 편차·처리 가능 K·오경보 허용 수준을 보고 실제 도입 여부를 판단한다.

소급 입력은 `recorded_at`이 있으면 기준일 이후 행을 제외한다. 이 필드가 없는 원본은 당시 실제 가용성까지 시스템이 확인할 수 없으므로 공급자가 시점 일관성을 보장해야 한다.
