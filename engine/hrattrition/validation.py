"""4단계 검증: 실제 검사로 상태를 산출한다(P0-1).

각 항목: id, item, status(pass|warn|fail|not_evaluable|not_applicable), detail, evidence(근거 ID 목록).
상수로 '통과'를 적지 않는다. 검사할 수 없으면 not_evaluable로 둔다.
"""
import numpy as np
import pandas as pd

from . import config as C
from .features import ALLOWED_FEATURES, FEATURE_DOCS, PLAIN
from .hypotheses import HYPOTHESES, _risk_z
from .model import heuristic_scores

STATUS_KO = {"pass": "통과", "warn": "경고", "fail": "실패", "not_evaluable": "평가 불가", "not_applicable": "해당 없음"}


def _check(cid, item, status, detail, evidence=None):
    return dict(id=cid, item=item, status=status, status_ko=STATUS_KO[status], detail=detail, evidence=evidence or [])


def _counterexamples(f: pd.DataFrame, s: pd.DataFrame, k: int = 3) -> list[dict]:
    out = []
    low, high = s.risk_band == "낮음", s.risk_band == "높음"
    any_strong = pd.Series(False, index=f.index)
    for hid, text, feats in HYPOTHESES:
        strong = pd.Series(False, index=f.index)
        for c in feats:
            if c in f and f[c].notna().any():
                strong |= (_risk_z(f, c) >= 1.5).fillna(False)
        any_strong |= strong
        ce = f[strong & low]
        if len(ce):
            out.append(dict(hypothesis=hid, type="강한 신호인데 밴드 낮음", count=int(len(ce)),
                            examples=[f"{i} {n}" for i, n in zip(ce.index[:k], ce["name"].iloc[:k])],
                            evidence=[f"EV-CE-{hid}-{i}" for i in ce.index[:k]],
                            reading="이 신호 하나로 위험을 단정하면 안 된다는 뜻. 다른 신호가 안전 쪽이었다."))
    hi_quiet = f[high & ~any_strong]
    if len(hi_quiet):
        out.append(dict(hypothesis="전체", type="밴드 높음인데 강한 단일 신호 없음", count=int(len(hi_quiet)),
                        examples=[f"{i} {n}" for i, n in zip(hi_quiet.index[:k], hi_quiet["name"].iloc[:k])],
                        evidence=[f"EV-CE-ALL-{i}" for i in hi_quiet.index[:k]],
                        reading="약한 신호가 여럿 겹친 경우. 면담 때 어느 신호인지 상세를 봐야 한다."))
    return out


def _sensitivity(features: pd.DataFrame, scores: pd.DataFrame, weights: dict | None) -> dict:
    base = scores.set_index("employee_id").risk_score
    rows = []
    for c in ALLOWED_FEATURES:
        if c == "tenure_years" or c not in features or features[c].isna().all():
            continue
        w = {**(weights or {}), c: 0.0}
        alt = heuristic_scores(features, w).scores.set_index("employee_id")
        rho = float(base.corr(alt.risk_score, method="spearman"))
        flips = int((alt.risk_band != scores.set_index("employee_id").risk_band).sum())
        rows.append(dict(signal=FEATURE_DOCS[c][0], feature=c, rank_corr_without=round(rho, 3), band_changes=flips))
    rows.sort(key=lambda r: r["rank_corr_without"])
    return dict(method="신호 하나를 가중치 0으로 두고 재계산 → 원래 지수와의 순위 상관·밴드 변경 수", most_influential=rows[:5], all=rows)


def validate(features: pd.DataFrame, res, labels: pd.Series | None, as_of: str, as_of_source: str | None,
             weights: dict | None, precheck_issues: list[dict] | None = None) -> dict:
    f = features.set_index("employee_id")
    s = res.scores.set_index("employee_id").reindex(f.index)
    checks = []

    # V1 기준일 출처
    if as_of and as_of_source:
        checks.append(_check("V1", "기준일 출처 확인", "pass", f"기준일 {as_of} ({'사용자 지정' if as_of_source == 'user' else as_of_source})", ["EV-ASOF"]))
    else:
        checks.append(_check("V1", "기준일 출처 확인", "fail", "기준일을 확인하지 못했다", []))

    # V2 기준일 이후 정보 제외(누수): 실제 검사한 날짜 컬럼이 있어야 평가 가능
    excluded = {k: int(v) for k, v in features.attrs.get("excluded_after_as_of", {}).items() if v}
    unparse = {k: int(v) for k, v in features.attrs.get("unparseable_dates", {}).items() if v}
    checked = features.attrs.get("date_columns_checked", [])
    if not checked:
        checks.append(_check("V2", "기준일 이후 정보 제외", "not_evaluable", "검사할 날짜 컬럼이 없다(이력 없는 명부). 시점 누수를 확인할 수 없다.", []))
    elif features.attrs.get("unchecked_date_tables"):
        checks.append(_check("V2", "기준일 이후 정보 제외", "not_evaluable", "날짜가 없어 제외한 이력: " + ", ".join(features.attrs["unchecked_date_tables"])))
    elif unparse:
        checks.append(_check("V2", "기준일 이후 정보 제외", "warn",
                             f"날짜 컬럼 {len(checked)}개 검사, 기준일 이후 {sum(excluded.values())}행 제외. 날짜를 읽지 못해 제외한 행 {sum(unparse.values())}건: " +
                             ", ".join(f"{k} {v}" for k, v in unparse.items()), [f"EV-DATE-{c}" for c in checked]))
    else:
        checks.append(_check("V2", "기준일 이후 정보 제외", "pass", f"날짜 컬럼 {len(checked)}개 검사, 기준일 이후 {sum(excluded.values())}행 제외",
                             [f"EV-DATE-{c}" for c in checked]))

    # V3 민감 속성 미사용: 피처 테이블과 실제 사용 컬럼을 검사
    used = set(res.metrics.get("features_used", []))
    leaked_table = [c for c in f.columns if c in C.SENSITIVE_COLUMNS or any(str(c).startswith(x + "_") for x in C.SENSITIVE_COLUMNS)]
    leaked_used = [c for c in used if c in C.SENSITIVE_COLUMNS or any(str(c).startswith(x + "_") for x in C.SENSITIVE_COLUMNS)]
    excluded_used = [c for c in used if c in C.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS]
    if leaked_table or leaked_used:
        checks.append(_check("V3", "민감 속성·제외 신호 미사용", "fail", f"민감 속성 발견: 피처 테이블 {leaked_table}, 점수 사용 {leaked_used}", []))
    elif excluded_used:
        checks.append(_check("V3", "민감 속성·제외 신호 미사용", "fail", f"개인 가중치 제외 신호가 점수에 사용됨: {excluded_used}", []))
    else:
        checks.append(_check("V3", "민감 속성·제외 신호 미사용", "pass", f"사용 신호 {len(used)}개 모두 허용 목록 안. 제외: {', '.join(C.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS)}", ["EV-POLICY"]))

    # V4 점수 산정 방식 공개
    if res.mode == "heuristic":
        method = dict(mode="heuristic", description="허용 신호의 표준화 z × 공개 가중치 합 + 근속 구간 가산 → 표준화 → 시그모이드(0~1). 회사 내 상대 지수.",
                      weights_used=res.metrics.get("weights"), tenure_bump=res.metrics.get("tenure_bump"), scale=res.metrics.get("scale"),
                      bands=res.metrics.get("bands"), band_counts=res.metrics.get("band_counts"))
        checks.append(_check("V4", "점수 산정 방식 공개", "pass", f"규칙 지수 · 가중치 {len(res.metrics.get('weights', {}))}개 · scale {res.metrics.get('scale')}", ["EV-METHOD"]))
    else:
        method = dict(mode="model", description=res.metrics.get("algorithm"), cv_auc=res.metrics.get("cv_auc"), positives=res.metrics.get("positives"),
                      n_train=res.metrics.get("n_train"), bands=res.metrics.get("bands"), band_counts=res.metrics.get("band_counts"))
        checks.append(_check("V4", "점수 산정 방식 공개", "pass", f"{res.metrics.get('algorithm')} · 고정 모델 {res.metrics.get('model_id', '실험용')} · {res.metrics.get('band_method', '평가용')}", ["EV-METHOD"]))

    # V5 반례
    ce = _counterexamples(f, s)
    checks.append(_check("V5", "반례 확인", "pass" if ce else "warn", f"{sum(c['count'] for c in ce)}건 기록" if ce else "반례가 하나도 없음: 지수가 단일 신호에 좌우되는지 V6에서 확인",
                         [e for c in ce for e in c["evidence"]]))

    # V6 순위 안정성(규칙 모드만)
    sens = None
    if leaked_table:
        checks.append(_check("V6", "순위 안정성", "not_evaluable", "민감 속성이 피처 테이블에 있어 재계산 기반 감도 분석을 생략(V3 실패 선결)", []))
    elif res.mode == "heuristic":
        sens = _sensitivity(features, res.scores, weights)
        if sens["most_influential"]:
            worst = sens["most_influential"][0]
            checks.append(_check("V6", "순위 안정성", "pass" if worst["rank_corr_without"] >= 0.8 else "warn",
                                 f"가장 영향 큰 신호 '{worst['signal']}' 제거 시 순위 상관 {worst['rank_corr_without']}, 밴드 변경 {worst['band_changes']}명", ["EV-SENS"]))
        else:
            checks.append(_check("V6", "순위 안정성", "not_evaluable", "사용 가능한 신호가 없어 감도 분석 불가", []))
    else:
        checks.append(_check("V6", "순위 안정성", "not_applicable", "고정 모델 적용: 기간별 성능과 직원별 그룹 평가는 evaluation.json 참조", []))

    # V7 커버리지
    coverage = [dict(signal=FEATURE_DOCS[c][0], feature=c, coverage_pct=round(100 * float(f[c].notna().mean()), 1)) for c in ALLOWED_FEATURES if c in f]
    low_cov = [r for r in coverage if r["coverage_pct"] < 50 and r["feature"] in used]
    zero = [r for r in low_cov if r["coverage_pct"] == 0]
    partial = [r for r in low_cov if r["coverage_pct"] > 0]
    detail = "사용 신호 모두 커버리지 50% 이상" if not low_cov else \
        (f"데이터 없음 {len(zero)}개" if zero else "") + ("; " if zero and partial else "") + \
        ("50% 미만: " + ", ".join(f"{r['signal']} {r['coverage_pct']}%" for r in partial[:5]) + (f" 외 {len(partial) - 5}개" if len(partial) > 5 else "") if partial else "")
    checks.append(_check("V7", "사용 신호 커버리지", "warn" if low_cov else "pass", detail, ["EV-COV"]))

    # V8 입력 선검사 결과 반영
    pre = precheck_issues or []
    warns = [i for i in pre if i["severity"] == "warn"]
    checks.append(_check("V8", "입력 선검사", "warn" if warns else "pass", f"경고 {len(warns)}건: " + "; ".join(i["code"] for i in warns) if warns else "차단·경고 이슈 없음",
                         [f"EV-PRE-{i['code']}" for i in pre]))

    # V9 라벨·정확도
    if res.metrics.get("evaluation"):
        ev = res.metrics["evaluation"]
        checks.append(_check("V9", "미래 기간 평가", "warn" if ev.get("synthetic") or not ev.get("beats_baseline") else "pass",
                             f"미래 기간 평균 Precision@{ev['top_k']}={ev['future_evaluation']['mean_precision_at_k']:.3f}; " + ("합성 데이터, 실제 성능 미검증" if ev.get("synthetic") else "운영 출시 판단은 별도"), ["evaluation.json"]))
    elif res.mode == "heuristic":
        checks.append(_check("V9", "정확도 검증", "not_evaluable", "진단 지수에는 미래 기간 예측 성능이 없다. 별도 학습·기간 분리 평가가 필요하다.", []))
    elif labels is None:
        checks.append(_check("V9", "정확도 검증", "not_evaluable", "실제 퇴사 라벨이 없어 정확도를 측정할 수 없다. 지수는 진단용이며 예측 확률이 아니다.", []))
    else:
        checks.append(_check("V9", "정확도 검증", "warn", f"무작위 층화 교차검증 AUC {res.metrics.get('cv_auc')} (양성 {res.metrics.get('positives')}). 시간 분리 평가가 아니므로 출시 성능 근거가 아니다(P5).", ["EV-METHOD"]))

    limitations = ["단면 분석: 기준일 하나의 상태로 계산했고 과거 시점과의 비교는 하지 않았다.",
                   "성별·국적·체류자격·혼인·가족·주거·거주지·병역은 쓰지 않았다. 다른 신호가 이들과 상관될 가능성은 남아 있다.",
                   "병가·설문 미응답은 개인 점수에 쓰지 않았다(조직 집계에만 참고)."]
    if res.mode == "heuristic":
        limitations.insert(0, "고정 예측 모델을 적용하지 않아 가중치는 가설에 근거한 사전 설정값이다. 지수는 예측 확률이 아니며 정확도를 측정할 수 없다.")
    if low_cov:
        limitations.append("커버리지 50% 미만 신호: " + ", ".join(r["signal"] for r in low_cov))
    if res.metrics.get("fallback_reason"):
        limitations.append(res.metrics["fallback_reason"])
    overall = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "pass"
    return dict(overall=overall, overall_ko=STATUS_KO[overall], checklist=checks,
                leakage=dict(as_of=as_of, as_of_source=as_of_source, date_columns_checked=checked, excluded_rows_by_table=excluded,
                             unparseable_rows_by_table=unparse, total_excluded=int(sum(excluded.values()))),
                counterexamples=ce, scoring_method=method, sensitivity=sens, coverage=coverage, limitations=limitations)
