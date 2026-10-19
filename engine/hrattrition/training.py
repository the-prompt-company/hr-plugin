"""Time-separated candidate selection, untouched future evaluation, frozen inference.

Input is a JSON manifest of historical snapshot files. All preprocessing is fit
inside training only. Validation chooses a candidate; test never chooses it.
The deployed bundle is that exact evaluated model, not a post-test refit.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import __version__
from .data import load_any
from .features import ALLOWED_FEATURES, FEATURE_DOCS, build_features, numeric_matrix
from .labels import load_labels
from .precheck import precheck, blocking
from .model import ScoreResult, policy_dict

CORE = ('tenure_years', 'grade_level', 'base_annual_krw10k', 'band_position',
        'months_since_raise', 'last_raise_pct', 'last_rating_score', 'rating_trend', 'ot_hours_m6', 'ot_trend')
SCHEMA = 'hr-frozen-model/1'


def metrics_at_k(y, scores, k):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    if int(k) <= 0:
        raise ValueError("K는 양수여야 합니다.")
    if len(y) == 0:
        raise ValueError('평가 표본이 없습니다.')
    actual_k = min(int(k), len(y))
    order = np.argsort(-scores, kind='stable')[:actual_k]
    hits, positives = int(y[order].sum()), int(y.sum())
    base = float(y.mean())
    bins = []
    for lo in np.arange(0, 1, .2):
        mask = (scores >= lo) & (scores < lo+.2 if lo < .8 else scores <= 1)
        if mask.any():
            bins.append(dict(n=int(mask.sum()), predicted=float(scores[mask].mean()), observed=float(y[mask].mean())))
    return dict(n=len(y), positives=positives, requested_k=int(k), k=actual_k, hits=hits,
                precision_at_k=hits/actual_k, recall_at_k=hits/positives if positives else None,
                lift_at_k=(hits/actual_k)/base if base else None, base_rate=base,
                roc_auc=float(roc_auc_score(y, scores)) if 0<positives<len(y) else None,
                average_precision=float(average_precision_score(y, scores)) if positives else None,
                brier=float(brier_score_loss(y, scores)), calibration_bins=bins)


def evaluate_periods(frame, scores, k):
    rows = []
    for date, g in frame.assign(prediction=scores).groupby('snapshot_date', sort=True):
        # Stable employee ordering breaks ties reproducibly.
        g = g.sort_values('employee_id', kind='stable')
        rows.append(dict(as_of=str(date.date()), **metrics_at_k(g.label, g.prediction, k)))
    precision = float(np.mean([r['precision_at_k'] for r in rows]))
    recall = [r['recall_at_k'] for r in rows if r['recall_at_k'] is not None]
    lift = [r['lift_at_k'] for r in rows if r['lift_at_k'] is not None]
    return dict(periods=rows, mean_precision_at_k=precision,
                mean_recall_at_k=float(np.mean(recall)) if recall else None,
                mean_lift_at_k=float(np.mean(lift)) if lift else None,
                pooled=metrics_at_k(frame.label, scores, k))


def _read_history(manifest_path):
    spec = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    target, horizon = spec.get('target', '자발적 퇴사'), int(spec.get('horizon_days', 90))
    rows, sources, counts = [], [], []
    company = None
    for item in spec.get('snapshots', []):
        source = (Path(manifest_path).parent / item['input']).resolve()
        ds = load_any(str(source), as_of=item.get('as_of'))
        if blocking(precheck(ds)):
            raise ValueError(f'{source.name}: 입력 선검사 실패')
        name = ds.manifest['company']['name']
        if company is not None and company != name:
            raise ValueError('서로 다른 회사의 스냅샷을 합칠 수 없습니다.')
        company = name
        f = build_features(ds)
        if f.attrs.get('unchecked_date_tables'):
            raise ValueError(f'시점 확인 불가 이력: {f.attrs["unchecked_date_tables"]}')
        label_path = (Path(manifest_path).parent/item['labels']).resolve() if item.get('labels') else None
        y = load_labels(ds, label_path, target, horizon)
        if y is None:
            raise ValueError(f'{source.name}: 사건 이력 라벨이 필요합니다.')
        f['label'] = f.employee_id.map(y)
        f['snapshot_date'] = pd.Timestamp(ds.as_of)
        f['label_end'] = pd.Timestamp(ds.as_of) + pd.Timedelta(days=horizon)
        counts.append(dict(file=source.name, eligible=len(f), excluded_labels=int(f.label.isna().sum())))
        rows.append(f[f.label.notna()])
        for path in [source] + ([label_path] if label_path else []):
            sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not rows:
        raise ValueError('과거 스냅샷이 없습니다.')
    frame = pd.concat(rows, ignore_index=True).sort_values(['snapshot_date', 'employee_id']).reset_index(drop=True)
    if frame.duplicated(['employee_id', 'snapshot_date']).any():
        raise ValueError('동일 직원·기준일이 중복되었습니다.')
    return spec, frame, company, sources, counts


def train(manifest_path, out_dir):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    import joblib
    spec, frame, company, sources, counts = _read_history(manifest_path)
    k, horizon = int(spec.get('top_k', 20)), int(spec.get('horizon_days', 90))
    if k <= 0:
        raise ValueError('top_k는 양수여야 합니다.')
    val_start, test_start = pd.Timestamp(spec['validation_start']), pd.Timestamp(spec['test_start'])
    observed_through = pd.Timestamp(spec['observed_through'])
    if not val_start < test_start < observed_through:
        raise ValueError('validation_start < test_start < observed_through 순서가 필요합니다.')
    train_rows = frame[(frame.snapshot_date < val_start) & (frame.label_end < val_start)].copy()
    val = frame[(frame.snapshot_date >= val_start) & (frame.snapshot_date < test_start) & (frame.label_end < test_start)].copy()
    test = frame[(frame.snapshot_date >= test_start) & (frame.label_end <= observed_through)].copy()
    for name, part in [('학습', train_rows), ('검증', val), ('미래 평가', test)]:
        if part.empty or part.label.nunique() < 2:
            raise ValueError(f'{name}: 성숙한 양성·음성 라벨이 모두 필요합니다. 기간과 관측 종료일을 확인하세요.')
    if train_rows.groupby('label').size().min() < 10:
        raise ValueError('학습에는 각 클래스 10개 이상이 필요합니다. 이는 성능 보장 기준이 아닙니다.')
    feature_sets = {'core': [c for c in CORE if c in frame], 'all': [c for c in ALLOWED_FEATURES if c in frame]}
    candidates, fitted = [], {}
    for family in ('baseline', 'logistic', 'hist_gradient_boosting'):
        for feature_set in (['core'] if family == 'baseline' else ['core', 'all']):
            cols = [c for c in feature_sets[feature_set] if train_rows[c].notna().any()]
            if not cols:
                continue
            if family == 'baseline':
                estimator = DummyClassifier(strategy='prior')
            elif family == 'logistic':
                estimator = LogisticRegression(C=.5, max_iter=2000, random_state=42)
            else:
                estimator = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=20,
                                                            l2_regularization=1, early_stopping=False, random_state=42)
            pipe = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('scale', StandardScaler()), ('clf', estimator)])
            Xtrain = train_rows[cols].astype(float)
            pipe.fit(Xtrain, train_rows.label.astype(int))
            pred = pipe.predict_proba(val[cols].astype(float))[:,1]
            key = f'{family}:{feature_set}'
            result = evaluate_periods(val, pred, k)
            candidates.append(dict(candidate=key, feature_count=len(cols), validation=result))
            fitted[key] = (pipe, cols)
    # No test observations or test metrics enter this selection.
    winner = sorted(candidates, key=lambda r: (-r['validation']['mean_precision_at_k'],
                    -r['validation']['pooled']['average_precision'], r['validation']['pooled']['brier'], r['candidate']))[0]
    key = winner['candidate']; pipe, cols = fitted[key]
    test_scores = pipe.predict_proba(test[cols].astype(float))[:,1]
    future = evaluate_periods(test, test_scores, k)
    baseline_key = 'baseline:core'
    bp, bc = fitted[baseline_key]
    baseline_future = evaluate_periods(test, bp.predict_proba(test[bc].astype(float))[:,1], k)
    train_ids = set(train_rows.employee_id)
    future_groups = {}
    for name, mask in [('previously_seen', test.employee_id.isin(train_ids)), ('new_employee', ~test.employee_id.isin(train_ids))]:
        if mask.any():
            future_groups[name] = evaluate_periods(test[mask], test_scores[mask.to_numpy()], k)
    features_changed = {}
    base_precision = future['mean_precision_at_k']
    # Validation-only permutation diagnostics; never a test-based feature selector.
    rng = np.random.default_rng(42)
    for col in cols:
        altered = val[cols].astype(float).copy()
        altered[col] = rng.permutation(altered[col].to_numpy())
        pm = evaluate_periods(val, pipe.predict_proba(altered)[:,1], k)
        features_changed[col] = winner['validation']['mean_precision_at_k'] - pm['mean_precision_at_k']
    version = hashlib.sha256(json.dumps(dict(spec=spec, sources=sources, candidate=key),sort_keys=True).encode()).hexdigest()[:16]
    report = dict(schema=SCHEMA, model_id=version, engine_version=__version__, company=company,
                  target=spec.get('target','자발적 퇴사'), horizon_days=horizon, top_k=k,
                  selection='validation mean Precision@K; AP and Brier tie-break; test untouched',
                  selected=key, candidates=candidates, future_evaluation=future, baseline_future=baseline_future,
                  employee_groups=future_groups, validation_permutation_precision_drop=features_changed,
                  split=dict(train_end=str(train_rows.snapshot_date.max().date()), validation_start=str(val_start.date()),
                             test_start=str(test_start.date()), observed_through=str(observed_through.date()),
                             train_rows=len(train_rows), validation_rows=len(val), test_rows=len(test),
                             excluded_boundary_rows=len(frame)-len(train_rows)-len(val)-len(test)),
                  label_exclusions=counts, source_hashes=sources, synthetic=bool(spec.get('synthetic', False)),
                  beats_baseline=base_precision>baseline_future['mean_precision_at_k'],
                  probability_status='uncalibrated',
                  limitations=['점수는 우선순위용이며 보정된 퇴사 확률이 아님.',
                               '반복 직원·겹치는 예측기간의 의존성이 있어 행을 독립 표본으로 해석하지 않음.',
                               '시간 평가 실시만으로 출시 성능이 보장되지 않음. 처리 용량 K와 오경보 허용 수준을 별도 검토.',
                               '합성 데이터는 처리 경로 검증 전용.' if spec.get('synthetic') else '실제 운영에서는 추가 미래 기간 성능을 추적해야 함.'])
    # Baseline training reference is kept for consistent inference explanations.
    medians = train_rows[cols].median().fillna(0).to_dict()
    bundle = dict(schema=SCHEMA, model=pipe, columns=cols, medians=medians, report=report,
                  trained_through=str(train_rows.label_end.max().date()),
                  available_after=str(observed_through.date()))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out/'model.joblib')
    (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame([dict(candidate=r['candidate'], features=r['feature_count'],
                       precision_at_k=r['validation']['mean_precision_at_k'],
                       recall_at_k=r['validation']['mean_recall_at_k'],
                       average_precision=r['validation']['pooled']['average_precision'],
                       brier=r['validation']['pooled']['brier']) for r in candidates]).to_csv(out/'candidate_comparison.csv',index=False,encoding='utf-8-sig')
    test.assign(prediction=test_scores)[['employee_id','snapshot_date','label','prediction']].to_csv(out/'future_predictions.csv',index=False,encoding='utf-8-sig')
    return out/'model.joblib', report


def predict_frozen(features, model_path, as_of, target, horizon_days, company):
    import joblib
    bundle = joblib.load(model_path)  # Only load locally generated/trusted model bundles.
    if bundle.get('schema') != SCHEMA:
        raise ValueError('지원하지 않는 모델 형식입니다. train 명령으로 생성하세요.')
    report = bundle['report']
    if report['target'] != target or report['horizon_days'] != horizon_days:
        raise ValueError('요청한 퇴사 유형·예측 기간이 학습 모델과 다릅니다. 일치하는 모델을 사용하세요.')
    if report['company'] != company:
        raise ValueError('학습 회사와 예측 회사가 다릅니다.')
    if pd.Timestamp(as_of) <= pd.Timestamp(bundle['available_after']):
        raise ValueError('현재 예측 기준일은 모델 선택·미래 평가의 관측 종료일 이후여야 합니다.')
    cols = bundle['columns']; missing = [c for c in cols if c not in features or features[c].isna().all()]
    if missing:
        raise ValueError(f'학습에 사용한 신호가 예측 입력에서 전부 누락되었습니다: {missing}')
    X = features.set_index('employee_id')[cols].astype(float)
    if (X.notna().mean(axis=1) < .5).any():
        raise ValueError('직원별 사용 신호 커버리지가 50% 미만입니다. 데이터를 보완하세요.')
    pipe = bundle['model']; pred = pipe.predict_proba(X)[:,1]
    # Difference when one observed feature is replaced by its training median.
    # Not causal effects and not additive SHAP; works for every candidate family.
    contrib = pd.DataFrame(index=X.index)
    for col in cols:
        altered = X.copy(); altered[col] = bundle['medians'][col]
        contrib[col] = pred - pipe.predict_proba(altered)[:,1]
    drivers = []
    from .features import hr_value
    for eid in X.index:
        ranked = contrib.loc[eid].sort_values(ascending=False)
        selected = [c for c in ranked.index if ranked[c] > .005][:3]
        drivers.append('; '.join(hr_value(c, X.at[eid,c], bundle['medians'][c]) for c in selected) or '뚜렷한 단일 신호 없음')
    order = pd.DataFrame({"employee_id": X.index, "score":pred}).sort_values(["score", "employee_id"],ascending=[False,True],kind="stable").employee_id
    rank = {eid:i+1 for i,eid in enumerate(order)}
    k = report['top_k']
    scores = pd.DataFrame(dict(employee_id=X.index, risk_score=np.round(pred,6),
                              risk_band=['높음' if rank[e]<=k else '주의' if rank[e]<=2*k else '낮음' for e in X.index],
                              top_drivers=drivers, drivers_plain=drivers, mode='model'))
    metrics = dict(mode='model', algorithm=report['selected'], model_id=report['model_id'],
                   features_used=cols, evaluation=report, probability_status='uncalibrated',
                   note='과거에 고정한 모델의 신규 직원 우선순위 점수. 퇴사 확률 아님.',
                   explanation_method='해당 값을 학습 중앙값으로 대체했을 때 점수 차이(인과 효과 아님)',
                   reference_medians=bundle['medians'], band_method=f'회사 내 상위 {k}명 높음, 다음 {k}명 주의',
                   band_counts=scores.risk_band.value_counts().to_dict(), **policy_dict())
    metrics["bands"] = {"높음": f"상위 {k}명", "주의": f"다음 {k}명", "낮음": "나머지"}
    return ScoreResult(scores, 'model', metrics, None, contrib)
