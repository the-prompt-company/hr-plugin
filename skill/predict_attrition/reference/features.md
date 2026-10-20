# 신호(피처) 목록과 입력 형식

## 표준 HR 워크북에서 쓰는 시트

| 신호 | 계산 | 필요한 시트 |
|---|---|---|
| 근속·나이·직급·보직·고용형태·계약 만료 임박·휴직 | 스냅샷 컬럼 | `v_employees_current` (없으면 `hr_employees`) |
| 밴드 내 위치, 마지막 인상 후 개월, 마지막 인상률 | 보상 이력의 마지막 유효 행 | `hr_compensation_history` |
| 승진 보류(2년) | 최근 2개 심사연도 '보류' 건수 | `hr_promotion_reviews` |
| 마지막 승진 후 개월, 전보 횟수 | 인사발령 | `hr_personnel_actions` |
| 최근 평가, 평가 추세, 자기평가−확정등급 | 확정 등급 S=5…D=1 | `hr_performance_reviews` |
| 월평균 연장(6개월), 연장 추세, 지각·결근·병가(6개월) | 월별 근태 집계 | `v_attendance_monthly` |
| 연차 사용률 | 올해 사용/발생 | `v_employees_current` |
| 휴가 반려(12개월) | 상태 '반려' | `hr_leave_requests` |
| 교육 시간(12개월) | 수료 과정 시간 합 | `hr_training_enrollments` |
| 온콜 주(12개월) | 온콜 담당 주 수 | `hr_oncall_shifts` |
| 설문 몰입·관리자·성장, 보상 인식, 워라밸, eNPS, 미응답 | 최근 설문 1개의 차원별 평균 | `hr_surveys`, `hr_survey_questions`, `hr_survey_answers`, `hr_survey_responses` |

사용 가능한 시트가 없으면 해당 신호는 원칙적으로 결측이 되고, 요약의 "계산하지 못한 신호"와 대시보드 설명에 표시된다. 규칙 가중치도 데이터가 있는 신호에만 적용된다.

라벨 시트는 과거 학습에서만 사용한다. 현재 예측에는 train으로 생성한 고정 모델이 필요하다. 형식은 `labels.md`.

## 직원 명부 한 장 모드: 컬럼 별칭

첫 시트가 직원 1인 1행이고 `hr_`/`v_` 접두어 시트가 없으면 이 모드다. 아래 별칭(대소문자·공백·괄호 무시)으로 컬럼을 인식한다.

| 표준 컬럼 | 별칭 예 | 만들어지는 신호 |
|---|---|---|
| employee_id | 사번, 사원번호, 직원ID | 식별자(없으면 자동 부여) |
| name | 이름, 성명 | 표시 |
| division_name / org_unit_name | 본부·사업부 / 팀·부서 | 그룹 집계 |
| grade_title / grade_level / role_title | 직급 / 직급레벨 / 직책 | 직급레벨, 보직자 |
| employment_type / employment_status | 고용형태 / 재직상태 | 기간제, 휴직 |
| hire_date / birth_date / age | 입사일 / 생년월일 / 나이 | 근속, 나이 |
| base_annual_krw10k | 연봉, 기준연봉(만원) | 기준연봉 |
| last_raise_date / last_raise_pct | 마지막 인상일 / 인상률 | 인상 후 개월, 인상률 |
| last_promotion_date / promotion_holds | 최근 승진일 / 승진 보류 | 승진 후 개월, 승진 보류 |
| last_rating / prev_rating | 최근 평가 / 직전 평가 (S~D) | 최근 평가, 평가 추세 |
| overtime_hours_monthly | 월평균 연장, 초과근무시간 | 월평균 연장 |
| late_count / absent_days / sick_days | 지각 / 결근 / 병가 | 근태 신호 |
| annual_leave_granted / annual_leave_used | 연차 발생 / 연차 사용 | 연차 사용률 |
| training_hours | 교육시간 | 교육 시간 |
| survey_engagement / survey_enps | 몰입도 / eNPS | 설문 신호 |
| contract_end_date | 계약 만료일 | 계약 만료 임박 |
| left | 퇴사여부 | 라벨(모델 모드) |

없는 컬럼의 신호는 빠진다. 신호가 너무 적으면(예: 근속·직급만) 지수가 거의 근속 구간만 반영하므로, 요약에서 그 한계를 말해야 한다.

## 제외하는 컬럼

성별, 국적, 체류자격, 혼인상태, 가족형태, 주거형태, 거주지역, 한국어 능력, 병역은 어떤 모드에서도 자동으로 쓰지 않는다.
