"""파이프라인 회귀 테스트 + P0 신뢰성 테스트. 외부 데이터 없이 tools/make_fixture.py로 워크북을 만들어 쓴다.

    python -m pytest tests -q
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hrattrition import __version__
from hrattrition import config as C
from hrattrition.dashboard import build_dashboard
from hrattrition.data import load_any, load_roster_frame
from hrattrition.features import ALLOWED_FEATURES, FEATURE_DOCS, PLAIN, build_features, numeric_matrix
from hrattrition.hypotheses import HYPOTHESES, evaluate_hypotheses
from hrattrition.labels import load_labels
from hrattrition.model import HEURISTIC_WEIGHTS, check_weights, heuristic_scores, model_scores, score
from hrattrition.overview import build_overview
from hrattrition.precheck import blocking, precheck
from hrattrition.profile import profile_dataset
from hrattrition.question import QuestionSpec, apply_population, parse_filters
from hrattrition.report import build_summary, load_spec, read_result_workbook, write_analysis_spec, write_result_workbook
from hrattrition.validation import validate

ROOT = Path(__file__).resolve().parents[1]
PREDICT = ROOT / "skill/predict_attrition/scripts/predict_attrition.py"
DASH = ROOT / "skill/make_dashboard/scripts/make_dashboard.py"
ENV = {**os.environ, "PYTHONUTF8": "1"}
N = 186
sys.path.insert(0, str(ROOT / "tools"))
from make_fixture import make  # noqa: E402


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, env=ENV)


def write_fixture(path: Path, with_labels: bool, seed: int = 7) -> Path:
    sheets = make(N, seed)
    if not with_labels:
        sheets.pop("퇴사라벨")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    return path


@pytest.fixture(scope="session")
def workbook(tmp_path_factory):
    """라벨 시트를 뺀 픽스처 워크북(진단 모드). 라벨이 필요한 테스트는 따로 만든다."""
    p = write_fixture(tmp_path_factory.mktemp("fx") / "hr_fixture_nolabels.xlsx", with_labels=False)
    ds = load_any(str(p))
    return ds, build_features(ds)


def _roster(n=4, with_id=True):
    df = pd.DataFrame({
        "이름": ["가", "나", "다", "라"][:n], "부서": ["영업", "영업", "개발", "개발"][:n], "직급": ["대리", "과장", "사원", "차장"][:n],
        "입사일": ["2024-03-01", "2018-01-02", "2025-09-01", "2012-05-01"][:n], "연봉(만원)": [4200, 6000, 3600, 8000][:n],
        "최근 평가": ["B", "A", "C", "S"][:n], "월평균 연장": [12, 4, 20, 2][:n], "성별": ["여", "남", "여", "남"][:n],
    })
    if with_id:
        df.insert(0, "사번", ["A1", "A2", "A3", "A4"][:n])
    return df


# ---------------- 회귀 ----------------

def test_workbook_loads_as_standard_dataset(workbook):
    ds, f = workbook
    assert ds.kind == "workbook" and ds.manifest["employee_count"] == len(f) == N
    assert ds.manifest["company"]["name"].startswith("(주)") and ds.manifest["as_of_source"].startswith("file:") and not ds.has("labels")
    assert ds.has("employees_current") and not ds.has("attendance_daily")
    assert f.attrs["unavailable"] == [] and f.attrs["has_history"]
    for k in FEATURE_DOCS:
        assert k in f.columns, k
    for col in C.SENSITIVE_COLUMNS:
        assert col not in f.columns
    assert f.band_position.dropna().between(0, 1).all()


def test_heuristic_weights_match_documented_direction_and_policy():
    for k, w in HEURISTIC_WEIGHTS.items():
        assert k in FEATURE_DOCS and k in PLAIN and k in ALLOWED_FEATURES
        assert FEATURE_DOCS[k][2] != 0 and np.sign(w) == FEATURE_DOCS[k][2], k
    for k in C.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS:
        assert k not in HEURISTIC_WEIGHTS and k not in ALLOWED_FEATURES


def test_heuristic_scores_schema_and_plain_drivers(workbook):
    _, f = workbook
    res = heuristic_scores(f)
    s = res.scores
    assert list(s.columns) == ["employee_id", "risk_score", "risk_band", "top_drivers", "drivers_plain", "mode"]
    assert s.risk_score.between(0, 1).all() and 0.05 * N <= (s.risk_band == "높음").sum() <= 0.3 * N
    assert set(res.metrics["features_used"]) <= set(ALLOWED_FEATURES)
    assert score(f, None, "auto").mode == "heuristic"


def test_model_mode_with_synthetic_labels(workbook):
    _, f = workbook
    X = numeric_matrix(f)
    rng = np.random.default_rng(0)
    signal = -X["survey_engagement"].fillna(X["survey_engagement"].mean()) + X["ot_hours_m6"].fillna(0) / 5
    p = 1 / (1 + np.exp(-(signal - signal.mean()) / signal.std()))
    y = pd.Series((rng.random(len(p)) < p).astype(int), index=X.index)
    y.iloc[: C.MIN_POSITIVES_FOR_MODEL] = 1
    res = model_scores(f, y)
    assert res.mode == "model" and 0 <= res.metrics["cv_auc"] <= 1 and res.metrics["coefficients"]["survey_engagement"] < 0


def test_roster_single_sheet_mode(tmp_path: Path):
    p = tmp_path / "roster.xlsx"
    _roster().to_excel(p, index=False)
    ds = load_any(str(p), as_of="2026-10-16")
    assert ds.kind == "roster" and ds.manifest["as_of_source"] == "user" and not ds.manifest["temp_ids"]
    f = build_features(ds)
    assert "sex" not in f.columns and f.last_rating_score.tolist() == [3, 4, 2, 5] and "survey_engagement" in f.attrs["unavailable"]
    res = heuristic_scores(f)
    summary = build_summary(f, res, ds.manifest, top_n=3)
    assert summary["n_employees"] == 4 and build_overview(f, ds.manifest["as_of"])["headcount"] == 4


def test_labels_from_sheet_and_csv(tmp_path: Path):
    roster = pd.DataFrame({"사번": [f"E{i}" for i in range(30)], "입사일": ["2020-01-01"] * 30, "퇴사여부": ["예"] * 12 + ["아니오"] * 18})
    ds = load_roster_frame(roster, "mem")
    with pytest.raises(ValueError, match="left"):
        load_labels(ds)
    p = tmp_path / "labels.csv"
    p.write_text("employee_id,left\nE0,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_labels(ds, p)


def test_result_workbook_roundtrip_and_dashboard(workbook, tmp_path: Path):
    ds, f = workbook
    res = heuristic_scores(f)
    summary = build_summary(f, res, ds.manifest, top_n=10)
    xlsx = write_result_workbook(f, res, summary, ds.manifest, tmp_path)
    f2, s2, meta = read_result_workbook(xlsx)
    assert len(f2) == N and meta["mode"] == "heuristic" and meta["engine_version"] == __version__ and meta["schema"] == C.RESULT_SCHEMA
    ov = build_overview(f2, meta["manifest"]["as_of"], ds)
    from hrattrition.report import read_result_extras
    extras = read_result_extras(xlsx)
    html = build_dashboard(f2, s2, meta["metrics"], meta["manifest"], ov, tmp_path / "d.html", meta["unavailable"], extras).read_text(encoding="utf-8")
    assert "조직별 HR 지표 히트맵" in html and "직원 목록" in html and "유지 우선순위" in html and "__DATA__" not in html and "__EXTRAS__" not in html
    assert "cdnjs" not in html and "<script src=" not in html   # 외부 리소스 없음(오프라인에서도 열림)
    json.loads(html[html.index("const DATA=") + 11: html.index(", DOCS=")])
    # 공유용: 이름·사번이 파일에서 제거되고 가명으로 바뀐다
    share = build_dashboard(f2, s2, meta["metrics"], meta["manifest"], ov, tmp_path / "s.html", meta["unavailable"], extras, share=True).read_text(encoding="utf-8")
    assert not any(n in share for n in f2["name"].head(20)) and not any(i in share for i in f2["employee_id"].head(20)) and "P-001" in share


# ---------------- 방법론 ----------------

def test_methodology_steps_produce_artifacts(workbook, tmp_path: Path):
    ds, f_all = workbook
    q = QuestionSpec.from_args(filters=parse_filters(["직군=개발직"]))
    q.as_of, q.as_of_source = ds.manifest["as_of"], ds.manifest["as_of_source"]
    assert "진단" in q.answerable_question() and "90일" in q.requested_question() and "예측하지 않는다" in q.answerable_question()
    f, pop = apply_population(f_all, q)
    assert 0 < pop["after"] < pop["before"] == N
    prof = profile_dataset(ds, f, set(f_all.employee_id))
    assert prof["employees"] == pop["after"] and any("모집단 제한" in i for i in prof["issues"])
    res = heuristic_scores(f)
    hyp = evaluate_hypotheses(f, res.scores, None)
    assert len(hyp["hypotheses"]) == len(HYPOTHESES) and all(h["status"].startswith("탐색적 증거") for h in hyp["hypotheses"] if h["signals"])
    val = validate(f, res, None, q.as_of, q.as_of_source, None, precheck(ds))
    by_id = {c["id"]: c["status"] for c in val["checklist"]}
    assert by_id["V1"] == "pass" and by_id["V2"] == "pass" and by_id["V3"] == "pass" and by_id["V9"] == "not_evaluable" and by_id["V6"] in ("pass", "warn")
    summary = build_summary(f, res, ds.manifest, 5, q.to_dict(), prof, hyp, val, pop)
    assert summary["analysis_mode"] == "diagnostic" and "예측 아님" in summary["mode_label"]
    xlsx = write_result_workbook(f, res, summary, ds.manifest, tmp_path)
    assert pd.ExcelFile(xlsx).sheet_names[:7] == ["1 질문 정의", "요약", "2 데이터 이해", "2 데이터 이슈", "3 가설", "4 검증", "5 팀별 확인사항"]
    spec = json.loads(write_analysis_spec(summary, res, None, "auto", tmp_path).read_text(encoding="utf-8"))
    assert spec["schema"] == C.SPEC_SCHEMA and spec["weights_full"] and spec["mode_applied"] == "heuristic" and spec["feature_policy"]["allowed_features"]


# ---------------- P0-1 검증 상태가 실제 검사로 산출된다 ----------------

def test_validation_flags_sensitive_feature_and_unparseable_dates(workbook):
    ds, f = workbook
    res = heuristic_scores(f)
    # 민감 컬럼이 피처 테이블에 섞이면 numeric_matrix가 거부하고, 검증은 fail
    f_bad = f.copy()
    f_bad.attrs = dict(f.attrs)
    f_bad["sex"] = "여"
    with pytest.raises(ValueError):
        numeric_matrix(f_bad)
    val = validate(f_bad, res, None, ds.manifest["as_of"], ds.manifest["as_of_source"], None)
    assert {c["id"]: c["status"] for c in val["checklist"]}["V3"] == "fail" and val["overall"] == "fail"
    # 날짜 변환 실패 행이 있으면 V2는 warn
    f_warn = f.copy()
    f_warn.attrs = {**f.attrs, "unparseable_dates": {"performance_reviews": 3}}
    val2 = validate(f_warn, res, None, ds.manifest["as_of"], ds.manifest["as_of_source"], None)
    assert {c["id"]: c["status"] for c in val2["checklist"]}["V2"] == "warn"


def test_validation_not_evaluable_without_date_columns(tmp_path: Path):
    df = _roster().drop(columns=["입사일"])
    p = tmp_path / "r.xlsx"
    df.to_excel(p, index=False)
    ds = load_any(str(p), as_of="2026-10-16")
    f = build_features(ds)
    res = heuristic_scores(f)
    val = validate(f, res, None, "2026-10-16", "user", None)
    assert {c["id"]: c["status"] for c in val["checklist"]}["V2"] == "not_evaluable"


def test_leakage_cutoff_excludes_future_and_unparseable_rows(workbook):
    ds, f0 = workbook
    base_excl = f0.attrs["excluded_after_as_of"].get("performance_reviews", 0)   # 픽스처 자체에 기준일 이후 평가가 들어 있다
    assert base_excl > 0
    ds2 = copy.copy(ds)
    ds2.tables = dict(ds.tables)
    rv = ds.tables["performance_reviews"].copy()
    fut = rv.head(3).copy()
    fut["finalized_at"] = pd.Timestamp(ds.as_of) + pd.Timedelta(days=30)
    bad = rv.head(2).copy()
    bad["finalized_at"] = "날짜아님"
    ds2.tables["performance_reviews"] = pd.concat([rv, fut, bad])
    f2 = build_features(ds2)
    assert f2.attrs["excluded_after_as_of"]["performance_reviews"] == base_excl + 3 and f2.attrs["unparseable_dates"]["performance_reviews"] == 2


# ---------------- P0-2 입력 선검사 ----------------

def test_precheck_blocks_missing_as_of_and_duplicate_ids(tmp_path: Path):
    p = tmp_path / "r.xlsx"
    _roster().to_excel(p, index=False)
    ds = load_any(str(p))   # --as-of 없음 → 오늘 날짜를 쓰지 않고 차단
    assert ds.manifest["as_of"] is None
    codes = {i["code"] for i in blocking(precheck(ds))}
    assert "AS_OF_MISSING" in codes
    with pytest.raises(ValueError):
        build_features(ds)
    dup = _roster()
    dup.loc[1, "사번"] = "A1"
    dup.to_excel(p, index=False)
    ds = load_any(str(p), as_of="2026-10-16")
    codes = {i["code"] for i in blocking(precheck(ds))}
    assert "ID_DUPLICATE" in codes
    r = run(PREDICT, "--input", p, "--out", tmp_path / "out", "--as-of", "2026-10-16")
    assert r.returncode == 2 and (tmp_path / "out" / "diagnostics.json").exists() and "Traceback" not in r.stderr
    diag = json.loads((tmp_path / "out" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diag["blocked"] and any(i["code"] == "ID_DUPLICATE" and i["resolution"] for i in diag["issues"])


def test_precheck_warns_on_temporary_ids(tmp_path: Path):
    p = tmp_path / "r.xlsx"
    _roster(with_id=False).to_excel(p, index=False)
    ds = load_any(str(p), as_of="2026-10-16")
    issues = precheck(ds)
    assert not blocking(issues) and any(i["code"] == "ID_TEMPORARY" and "comparison" in i["blocking_capabilities"] for i in issues)


def test_workbook_precheck_passes(workbook):
    ds, _ = workbook
    issues = precheck(ds)
    assert not blocking(issues)


# ---------------- P0-3 피처 정책 ----------------

def test_feature_policy_blocks_sensitive_and_excluded_weights():
    with pytest.raises(ValueError):
        check_weights({"sex": 1.0})
    with pytest.raises(ValueError):
        check_weights({"nationality_베트남": 1.0})
    with pytest.raises(ValueError):
        check_weights({"sick_days_m6": 0.3})
    with pytest.raises(ValueError):
        check_weights({"made_up": 1.0})
    assert check_weights({"survey_engagement": -1.5}) == {"survey_engagement": -1.5}


# ---------------- P0-4 재현, P0-5 버전, P0-6 표현 ----------------

def test_spec_reproduces_scores_and_cli_overrides_spec(workbook, tmp_path: Path):
    ds, _ = workbook
    out1, out2, out3 = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    r = run(PREDICT, "--input", ds.source, "--out", out1, "--horizon-days", "180", "--prompt", "누가 그만둘 것 같은지 봐줘")
    assert r.returncode == 0, r.stderr
    assert all(tag in r.stdout for tag in ("[2 선검사]", "[1 질문]", "[2 데이터]", "[3 가설]", "[4 검증]", "[5 행동]", "[6 재사용]"))
    s1 = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))
    assert s1["question"]["horizon_days"] == 180 and s1["analysis_mode"] == "diagnostic" and "예측 아님" in s1["mode_label"]
    assert "퇴사 확률" not in json.dumps(s1["question"], ensure_ascii=False) and "예측하지 않는다" in s1["question"]["answerable_question"]
    assert (out1 / "run_manifest.json").exists() and json.loads((out1 / "run_manifest.json").read_text(encoding="utf-8"))["engine_version"] == __version__
    # 스펙만으로 재실행 → 점수·밴드 동일
    r = run(PREDICT, "--input", ds.source, "--out", out2, "--spec", out1 / "analysis_spec.json")
    assert r.returncode == 0, r.stderr
    a, b = pd.read_csv(out1 / "scores.csv"), pd.read_csv(out2 / "scores.csv")
    assert a[["employee_id", "risk_score", "risk_band"]].equals(b[["employee_id", "risk_score", "risk_band"]])
    assert json.loads((out2 / "summary.json").read_text(encoding="utf-8"))["question"]["horizon_days"] == 180
    # 명시적 CLI가 스펙보다 우선
    r = run(PREDICT, "--input", ds.source, "--out", out3, "--spec", out1 / "analysis_spec.json", "--horizon-days", "90")
    assert r.returncode == 0, r.stderr
    assert json.loads((out3 / "summary.json").read_text(encoding="utf-8"))["question"]["horizon_days"] == 90


def test_spec_version_check(tmp_path: Path):
    bad = tmp_path / "spec.json"
    bad.write_text(json.dumps({"schema": C.SPEC_SCHEMA, "engine_version": "9.9.0"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_spec(bad)
    bad.write_text(json.dumps({"schema": "something/else"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_spec(bad)


def test_version_single_source():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert plugin["version"] == __version__
    assert "dynamic" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_fixture_requires_training_before_prediction(tmp_path: Path):
    """A labeled current file must not silently train and score the same employees."""
    p = write_fixture(tmp_path / "fx.xlsx", with_labels=True)
    r = run(PREDICT, "--input", p, "--out", tmp_path / "blocked")
    assert r.returncode == 2 and "train" in r.stderr and "Traceback" not in r.stderr
    r = run(PREDICT, "--input", p, "--mode", "heuristic", "--out", tmp_path / "diagnostic")
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "diagnostic" / "summary.json").read_text())
    assert s["analysis_mode"] == "diagnostic"
    assert s["validation"]["leakage"]["excluded_rows_by_table"]["personnel_actions"] > 0


def test_entry_scripts_run_end_to_end(workbook, tmp_path: Path):
    ds, _ = workbook
    r = run(PREDICT, "--input", ds.source, "--out", tmp_path)
    assert r.returncode == 0, r.stderr
    r = run(DASH, "--result", tmp_path / C.RESULT_FILENAME, "--data", ds.source, "--out", tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "dashboard.html").exists() and "원본 포함" in r.stdout
