"""P0-2 입력 선검사: 피처를 만들기 전에 기준일·사번·필수 구조·날짜 해석을 검사한다.

결과는 QualityIssue 목록이다. severity가 'blocking'이면 피처 계산을 멈추고 진단만 돌려준다.
  QualityIssue: code, severity(blocking|warn|info), message, affected_fields, source_refs, blocking_capabilities, resolution
"""
import re

import pandas as pd

from .data import Dataset

DATE_COLS = {"compensation_history": ["effective_from"], "promotion_reviews": ["decided_at"], "personnel_actions": ["effective_date"],
             "performance_reviews": ["finalized_at"], "leave_requests": ["requested_at"], "training_enrollments": ["completed_at"],
             "oncall_shifts": ["week_start"], "surveys": ["closed_at"], "roster": ["hire_date", "last_raise_date", "last_promotion_date", "contract_end_date"]}


def _issue(code, severity, message, affected=None, refs=None, blocks=None, resolution=""):
    return dict(code=code, severity=severity, message=message, affected_fields=affected or [], source_refs=refs or [],
                blocking_capabilities=blocks or [], resolution=resolution)


def precheck(ds: Dataset) -> list[dict]:
    issues: list[dict] = []
    src = ds.manifest.get("source_file") or ds.source

    # ---- 필수 구조 ----
    emp_table = next((t for t in ("employees_current", "employees", "roster") if ds.has(t)), None)
    if emp_table is None:
        issues.append(_issue("STRUCTURE_MISSING", "blocking", "직원 테이블(v_employees_current / hr_employees / 직원 명부 첫 시트)이 없다.",
                             blocks=["all"], resolution="직원 1인 1행 시트를 포함한 파일을 다시 올린다."))
        return issues
    emp = ds[emp_table]
    if "employee_id" not in emp:
        issues.append(_issue("ID_MISSING", "blocking", f"{emp_table}에 사번 컬럼이 없다.", affected=["employee_id"], refs=[f"{src}#{emp_table}"],
                             blocks=["all"], resolution="사번(employee_id) 컬럼을 추가하거나 어떤 컬럼이 사번인지 알려 준다."))
        return issues

    # ---- 기준일 ----
    as_of, as_of_src = ds.manifest.get("as_of"), ds.manifest.get("as_of_source")
    if not as_of:
        issues.append(_issue("AS_OF_MISSING", "blocking", "기준일을 파일에서 찾지 못했다(개요 시트 '기준일' 또는 employees_current.as_of).",
                             affected=["as_of"], refs=[src], blocks=["features", "scores", "comparison"],
                             resolution="--as-of YYYY-MM-DD로 기준일을 지정한다. 오늘 날짜를 자동으로 쓰지 않는다."))
    elif ds.manifest.get("as_of_conflict"):
        issues.append(_issue("AS_OF_CONFLICT", "blocking", f"파일 안의 기준일이 서로 다르다: {ds.manifest['as_of_conflict']}",
                             affected=["as_of"], refs=[src], blocks=["features", "scores"], resolution="--as-of로 어느 기준일을 쓸지 지정한다."))

    # ---- 사번 ----
    ids = emp["employee_id"].astype(str).str.strip()
    dup = ids[ids.duplicated(keep=False)]
    if len(dup):
        issues.append(_issue("ID_DUPLICATE", "blocking", f"{emp_table}에 중복 사번 {dup.nunique()}개({len(dup)}행): {', '.join(dup.unique()[:5])}",
                             affected=["employee_id"], refs=[f"{src}#{emp_table}"], blocks=["features", "scores"],
                             resolution="중복 행을 합치거나 어느 행이 맞는지 정한다. 이름이 같아도 자동으로 합치지 않는다."))
    if ids.isna().any() or (ids == "").any() or (ids.str.lower().isin(["nan", "none", "nat"])).any():
        n = int(ids.isna().sum() + (ids == "").sum() + (ids.str.lower().isin(["nan", "none", "nat"])).sum())
        issues.append(_issue("ID_BLANK", "blocking", f"{emp_table}에 사번이 비어 있는 행 {n}건", affected=["employee_id"], refs=[f"{src}#{emp_table}"],
                             blocks=["features", "scores"], resolution="사번을 채우거나 해당 행을 제거한다."))
    if ds.manifest.get("temp_ids"):
        issues.append(_issue("ID_TEMPORARY", "warn", "사번 컬럼이 없어 행 순서로 임시 ID를 만들었다. 이번 단면 분석에만 쓸 수 있고 다른 파일·시점과 연결할 수 없다.",
                             affected=["employee_id"], refs=[f"{src}#{emp_table}"], blocks=["comparison", "state"],
                             resolution="다음 파일부터 안정적인 사번 컬럼을 포함한다."))
    if ids.str.fullmatch(r"0\d+").any():
        issues.append(_issue("ID_LEADING_ZERO", "info", "선행 0이 있는 사번이 있다. 문자열로 유지했다(예: " + ids[ids.str.fullmatch(r"0\d+")].iloc[0] + ").",
                             affected=["employee_id"], resolution="다른 파일에서도 숫자로 변환되지 않았는지 확인한다."))
    if emp_table == "roster" and (ds.manifest.get("column_mapping") or {}) == {}:
        issues.append(_issue("MAPPING_EMPTY", "blocking", "직원 명부의 컬럼을 하나도 인식하지 못했다.", blocks=["all"],
                             resolution="컬럼명을 별칭 표(features.md)에 맞추거나 매핑을 알려 준다."))

    # ---- 날짜 해석 ----
    for table, cols in DATE_COLS.items():
        if not ds.has(table):
            continue
        df = ds[table]
        for c in cols:
            if c not in df:
                continue
            raw = df[c]
            parsed = pd.to_datetime(raw, errors="coerce")
            bad = int((parsed.isna() & raw.notna()).sum()) + df.attrs.get("date_parse_errors", {}).get(c, 0)
            if bad:
                issues.append(_issue("DATE_PARSE_FAIL", "warn", f"{table}.{c}에서 날짜로 읽지 못한 값 {bad}건(해당 행은 피처 계산에서 제외).",
                                     affected=[f"{table}.{c}"], refs=[f"{src}#{table}"], blocks=[], resolution="날짜 형식(YYYY-MM-DD)을 통일한다."))
            if as_of and (parsed > pd.Timestamp(as_of)).any():
                n = int((parsed > pd.Timestamp(as_of)).sum())
                issues.append(_issue("DATE_AFTER_AS_OF", "info", f"{table}.{c}에 기준일 이후 기록 {n}건(피처 계산에서 제외).",
                                     affected=[f"{table}.{c}"], refs=[f"{src}#{table}"]))
    if ds.has("attendance_monthly"):
        m = ds["attendance_monthly"].get("pay_month")
        if m is not None:
            ok = m.astype(str).str.fullmatch(r"\d{4}-\d{2}")
            if (~ok).any():
                issues.append(_issue("DATE_PARSE_FAIL", "warn", f"attendance_monthly.pay_month 형식 오류 {int((~ok).sum())}건(YYYY-MM이어야 함).",
                                     affected=["attendance_monthly.pay_month"], resolution="YYYY-MM으로 통일한다."))
    return issues


def blocking(issues: list[dict]) -> list[dict]:
    return [i for i in issues if i["severity"] == "blocking"]


def summarize(issues: list[dict]) -> str:
    if not issues:
        return "이슈 없음"
    counts = {}
    for i in issues:
        counts[i["severity"]] = counts.get(i["severity"], 0) + 1
    return ", ".join(f"{k} {v}" for k, v in counts.items())
