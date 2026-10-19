"""위험 점수: 라벨이 있으면 로지스틱 회귀(교차검증), 없으면 투명한 규칙 기반 지수.

두 모드 모두 결과 스키마가 같다:
  employee_id, risk_score(0~1), risk_band, top_drivers(기술적 표기), drivers_plain(HR용 평이한 표기), mode
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURE_DOCS, numeric_matrix, phrase_for, hr_value

# 규칙 가중치(표준화 z값에 곱한다). 부호는 FEATURE_DOCS의 위험 방향과 같아야 한다(테스트로 확인).
HEURISTIC_WEIGHTS = {
    "band_position": -1.0, "months_since_raise": 0.6, "last_raise_pct": -0.5,
    "promotion_holds_2y": 0.9, "months_since_promotion": 0.4,
    "last_rating_score": -0.8, "rating_trend": -0.5, "self_manager_gap": 0.6,
    "ot_hours_m6": 0.6, "ot_trend": 0.6, "late_count_m6": 0.5, "absent_days_m6": 0.5,
    "leave_rejected_12m": 0.4, "training_hours_12m": -0.3, "oncall_weeks_12m": 0.3,
    "survey_engagement": -1.0, "survey_pay_perception": -0.7, "survey_workload": -0.6, "survey_enps": -0.8,
    "is_fixed_term": 0.5, "contract_ending_90d": 0.8, "grade_level": -0.2,
}   # 병가(sick_days_m6)·설문 미응답(survey_nonresponse)은 개인 가중치에서 제외(config.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS)
TENURE_BUMP = {(0.0, 1.0): 0.4, (1.0, 3.0): 0.6, (3.0, 6.0): 0.2}   # 근속 구간별 가산(z 단위)
SCALE = 0.8   # 시그모이드 계수: '높음'(≥0.7)이 대략 상위 15%, '주의'(≥0.4)가 다음 30% 안팎


def check_weights(weights: dict | None) -> dict:
    """가중치 override 정책 검사: 허용 신호만, 민감 속성·개인 제외 신호는 어떤 경로로도 불가."""
    if not weights:
        return {}
    bad_sensitive = [k for k in weights if k in C.SENSITIVE_COLUMNS or any(k.startswith(s + "_") for s in C.SENSITIVE_COLUMNS)]
    bad_excluded = [k for k in weights if k in C.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS]
    unknown = [k for k in weights if k not in FEATURE_DOCS and k not in bad_sensitive]
    if bad_sensitive:
        raise ValueError(f"민감 속성은 가중치로 쓸 수 없습니다: {bad_sensitive}")
    if bad_excluded:
        raise ValueError(f"개인 위험 가중치에서 제외된 신호입니다(조직 집계에만 사용): {bad_excluded}")
    if unknown:
        raise ValueError(f"알 수 없는 신호: {unknown}. 허용 신호: {sorted(k for k in FEATURE_DOCS if k not in C.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS)}")
    return {k: float(v) for k, v in weights.items()}


def policy_dict() -> dict:
    return dict(allowed_features=[k for k in FEATURE_DOCS if k not in C.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS],
                excluded_from_individual_weights=list(C.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS), sensitive_never_used=list(C.SENSITIVE_COLUMNS),
                tenure_bump={f"{lo}-{hi}": v for (lo, hi), v in TENURE_BUMP.items()}, scale=SCALE, bands={b: lo for b, lo in C.RISK_BANDS})


@dataclass
class ScoreResult:
    scores: pd.DataFrame
    mode: str
    metrics: dict = field(default_factory=dict)
    model: object = None
    contrib: pd.DataFrame | None = None


def _zscores(X: pd.DataFrame) -> pd.DataFrame:
    mu, sd = X.mean(), X.std(ddof=0).replace(0, np.nan)
    return ((X - mu) / sd).fillna(0.0).clip(-3, 3)


def _strength(v: float) -> str:
    return "강함" if v >= 1.5 else "뚜렷" if v >= 0.8 else "약함"


def _drivers(contrib: pd.DataFrame, Z: pd.DataFrame, k: int = 3) -> tuple[pd.Series, pd.Series]:
    """위험 기여(contrib>0)가 큰 신호 상위 k. 평이한 표현은 기여 부호가 아니라 값(z)의 방향으로 고른다."""
    def tech(row):
        pos = row[row > 0.05].sort_values(ascending=False).head(k)
        return "; ".join(f"{FEATURE_DOCS.get(c, (c,))[0]}(+{v:.1f})" for c, v in pos.items()) or "뚜렷한 요인 없음"

    def plain(row):
        pos = row[row > 0.05].sort_values(ascending=False).head(k)
        parts = []
        for c, v in pos.items():
            z = float(Z.at[row.name, c]) if (c in Z.columns and row.name in Z.index) else 0.0
            parts.append(f"{phrase_for(c, z)}({_strength(v)})")
        return "; ".join(parts) or "뚜렷한 신호 없음"
    return contrib.apply(tech, axis=1), contrib.apply(plain, axis=1)


def _frame(X, score, contrib, mode, Z) -> pd.DataFrame:
    tech, plain = _drivers(contrib, Z)
    return pd.DataFrame({"employee_id": X.index, "risk_score": np.round(score, 3),
                         "risk_band": [C.risk_band(s) for s in score], "top_drivers": tech.to_numpy(),
                         "drivers_plain": plain.to_numpy(), "mode": mode})


def heuristic_scores(features: pd.DataFrame, weights: dict | None = None) -> ScoreResult:
    weights = {**HEURISTIC_WEIGHTS, **check_weights(weights)}
    X = numeric_matrix(features)
    available = [c for c in X.columns if X[c].notna().any()]
    Z = _zscores(X[available])
    w = pd.Series({k: v for k, v in weights.items() if k in Z.columns})
    contrib = Z[w.index] * w
    tenure = X["tenure_years"] if "tenure_years" in X else pd.Series(np.nan, index=X.index)
    bump = pd.Series(0.0, index=X.index)
    for (lo, hi), v in TENURE_BUMP.items():
        bump[(tenure >= lo) & (tenure < hi)] = v
    contrib["tenure_years"] = bump
    raw = contrib.sum(axis=1)
    sd = max(float(raw.std(ddof=0)), 1e-9)
    score = 1 / (1 + np.exp(-(raw / sd) * SCALE))
    Zall = _zscores(X[available])
    out = _frame(X, score.to_numpy(), contrib, "heuristic", Zall)
    observed = []
    for eid in X.index:
        ranked = contrib.loc[eid].sort_values(ascending=False)
        keys = [c for c in ranked.index if ranked[c] > .05][:3]
        observed.append("; ".join(hr_value(c, X.at[eid,c], X[c].median()) for c in keys) or "뚜렷한 단일 신호 없음")
    out["drivers_plain"] = observed
    used = sorted(w.index.tolist())
    skipped = sorted(set(weights) - set(used))
    metrics = dict(mode="heuristic",
                   note="고정 예측 모델을 적용하지 않아 규칙 기반 신호 지수를 계산했다. 예측 확률이 아니라 회사 안에서의 상대 지수(0~1)다.",
                   weights={k: weights[k] for k in used}, weights_skipped_no_data=skipped, weights_full=dict(weights),
                   features_used=sorted(X.columns.tolist()), **policy_dict(),
                   n_employees=int(len(out)), band_counts=out.risk_band.value_counts().to_dict())
    return ScoreResult(out, "heuristic", metrics, None, contrib)


def model_scores(features: pd.DataFrame, labels: pd.Series) -> ScoreResult:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = numeric_matrix(features)
    X = X[[c for c in X.columns if X[c].notna().any()]]
    y = labels.reindex(X.index)
    train_mask = y.notna()
    Xt, yt = X[train_mask], y[train_mask].astype(int)
    pos, neg = int(yt.sum()), int(len(yt) - yt.sum())
    if pos < C.MIN_POSITIVES_FOR_MODEL or neg < C.MIN_POSITIVES_FOR_MODEL:
        raise ValueError(f"모델 학습에는 양성·음성 각 {C.MIN_POSITIVES_FOR_MODEL}명 이상이 필요합니다 (양성 {pos}, 음성 {neg}).")
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000))])
    folds = min(5, pos, neg)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_prob = cross_val_predict(pipe, Xt, yt, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(yt, cv_prob))
    pipe.fit(Xt, yt)
    prob = pipe.predict_proba(X)[:, 1]
    coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=X.columns)
    Xs = pd.DataFrame(pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X)), index=X.index, columns=X.columns)
    contrib = Xs * coef
    out = _frame(X, prob, contrib, "model", Xs)
    metrics = dict(mode="model", algorithm="LogisticRegression(class_weight=balanced, C=0.5) + median impute + standardize",
                   n_train=int(len(yt)), positives=pos, cv_folds=folds, cv_auc=round(auc, 3),
                   coefficients={k: round(float(v), 3) for k, v in coef.sort_values(key=abs, ascending=False).items()},
                   features_used=sorted(X.columns.tolist()), **policy_dict(),
                   band_counts=out.risk_band.value_counts().to_dict())
    return ScoreResult(out, "model", metrics, pipe, contrib)


def score(features: pd.DataFrame, labels: pd.Series | None, mode: str = "auto", weights: dict | None = None) -> ScoreResult:
    if mode == "heuristic" or (mode == "auto" and labels is None):
        return heuristic_scores(features, weights)
    if labels is None:
        raise ValueError("모델 모드에는 라벨이 필요합니다.")
    raise ValueError("현재 입력으로 재학습하지 않습니다. train과 --model-artifact를 사용하세요.")


def save_result(res: ScoreResult, out_dir: Path):
    res.scores.to_csv(out_dir / "scores.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics.json").write_text(json.dumps(res.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if res.model is not None:
        try:
            import joblib
            joblib.dump(res.model, out_dir / "model.joblib")
        except Exception:  # noqa: BLE001 - 모델 저장은 선택 사항
            pass
