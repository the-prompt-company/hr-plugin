"""직원별 피처(신호) 테이블. 있는 테이블만으로 계산하고, 못 만든 신호는 결측으로 두고 목록에 남긴다.

FEATURE_DOCS: 신호명 → (라벨, 계산 설명, 위험 방향 +1/-1/0)
PLAIN: 신호명 → (위험 방향으로 치우쳤을 때의 평이한 표현, HR 확인 질문)
"""
import numpy as np
import pandas as pd

from . import config as C
from .data import Dataset

FEATURE_DOCS = {
    "tenure_years": ("근속(년)", "입사일~기준일", 0),
    "age": ("나이", "기준일 만 나이", 0),
    "grade_level": ("직급레벨", "1~6 일반, 7~10 임원", -1),
    "is_leader": ("보직자", "팀장·반장·연구소장·임원이면 1", -1),
    "is_fixed_term": ("기간제 계약", "기간제(계약직)이면 1", +1),
    "contract_ending_90d": ("계약 만료 임박", "계약 만료가 기준일 90일 이내면 1", +1),
    "on_leave": ("휴직 중", "기준일 휴직이면 1", 0),
    "base_annual_krw10k": ("기준연봉(만원)", "현재 유효 보상", -1),
    "band_position": ("밴드 내 위치", "(연봉−밴드 하한)/(상한−하한), 0~1", -1),
    "months_since_raise": ("마지막 인상 후 개월", "마지막 보상 변경 효력일 이후 개월 수", +1),
    "last_raise_pct": ("마지막 인상률(%)", "가장 최근 보상 변경률", -1),
    "promotion_holds_2y": ("승진 보류(2년)", "최근 2개 심사연도 '보류' 건수", +1),
    "months_since_promotion": ("마지막 승진 후 개월", "승진·선임 발령 없으면 입사 후 개월", +1),
    "transfers_count": ("전보 횟수", "인사발령 '전보' 건수", 0),
    "last_rating_score": ("최근 평가(1~5)", "S=5 … D=1, 미평가는 결측", -1),
    "rating_trend": ("평가 추세", "최근 등급 − 직전 등급", -1),
    "self_manager_gap": ("자기평가−확정등급", "최근 평가에서 self_score − 등급 점수", +1),
    "ot_hours_m6": ("월평균 연장(h, 6개월)", "최근 6개월 월평균 연장근로", +1),
    "ot_trend": ("연장근로 추세(h)", "최근 3개월 평균 − 그 전 3개월 평균", +1),
    "late_count_m6": ("지각 횟수(6개월)", "최근 6개월 합", +1),
    "absent_days_m6": ("결근(6개월)", "최근 6개월 합", +1),
    "sick_days_m6": ("병가(6개월)", "최근 6개월 합", +1),
    "leave_usage_rate": ("연차 사용률", "올해 사용/발생", 0),
    "leave_rejected_12m": ("휴가 반려(12개월)", "반려 상태 신청 수", +1),
    "training_hours_12m": ("교육 시간(12개월)", "수료 과정 시간 합", -1),
    "oncall_weeks_12m": ("온콜 주(12개월)", "온콜 담당 주 수", +1),
    "survey_engagement": ("설문 몰입·관리자·성장(1~5)", "최근 설문의 몰입·성장·관리자·자율성 문항 평균", -1),
    "survey_pay_perception": ("설문 보상 인식(1~5)", "최근 설문 '보상 인식' 문항", -1),
    "survey_workload": ("설문 워라밸(1~5)", "최근 설문 '워라밸' 문항", -1),
    "survey_enps": ("추천 의향(0~10)", "최근 설문 추천의향", -1),
    "survey_nonresponse": ("설문 미응답", "최근 설문 미응답이면 1", +1),
}
NUMERIC_FEATURES = list(FEATURE_DOCS)
CATEGORICAL_INFO = ["division_name", "org_unit_name", "job_family", "grade_title", "role_title", "work_schedule",
                    "employment_type", "employment_status", "hire_type", "recruit_channel", "education"]

# 평이한 표현(위험 방향으로 치우친 경우)과 확인 질문
PLAIN = {
    "grade_level": ("직급이 낮은 편", "경력 경로와 승진 기대가 맞는지"),
    "is_leader": ("보직 없음", "리더십 기회·역할 확장 여지가 있는지"),
    "is_fixed_term": ("기간제 계약", "정규직 전환 계획이 공유됐는지"),
    "contract_ending_90d": ("계약 만료 90일 이내", "연장·전환 의사 확인이 됐는지"),
    "base_annual_krw10k": ("연봉이 낮은 편", "동 직급 대비 보상 수준"),
    "band_position": ("보상 밴드 하단", "밴드 내 위치 조정 여지"),
    "months_since_raise": ("오랫동안 인상 없음", "마지막 보상 조정 시점과 사유"),
    "last_raise_pct": ("마지막 인상률 낮음", "인상 폭에 대한 본인 인식"),
    "promotion_holds_2y": ("승진 보류 반복", "보류 사유가 설명됐는지, 다음 심사 계획"),
    "months_since_promotion": ("오랫동안 승진 없음", "성장 정체감 여부"),
    "last_rating_score": ("최근 평가 낮음", "피드백·개선 지원이 있었는지"),
    "rating_trend": ("평가 하락", "하락 원인(업무·관계·건강)"),
    "self_manager_gap": ("자기평가와 확정등급 차이 큼", "평가 결과에 대한 수용도"),
    "ot_hours_m6": ("연장근로 많음", "업무량·인력 배치"),
    "ot_trend": ("연장근로 증가 추세", "최근 업무 변화·마감 압박"),
    "late_count_m6": ("지각 잦음", "출퇴근·건강·동기 변화"),
    "absent_days_m6": ("결근 있음", "개인 사정·건강"),
    "sick_days_m6": ("병가 많음", "건강 상태·업무 스트레스"),
    "leave_rejected_12m": ("휴가 반려 경험", "휴가 승인 관행·대체 인력"),
    "training_hours_12m": ("교육 참여 적음", "성장 기회 제공 여부"),
    "oncall_weeks_12m": ("온콜 부담 큼", "온콜 순번 분배"),
    "survey_engagement": ("설문 몰입도 낮음", "직무·관리자·성장 중 어디가 낮은지"),
    "survey_pay_perception": ("보상이 적절치 않다고 응답", "보상 인식과 실제 밴드 위치 비교"),
    "survey_workload": ("워라밸 응답 낮음", "업무량·근무시간"),
    "survey_enps": ("회사 추천 의향 낮음", "조직 문화·방향성 공감"),
    "survey_nonresponse": ("설문 미응답", "조직 소통 참여도"),
    "tenure_years": ("이직이 잦은 근속 구간(1~3년)", "초기 적응·경력 계획"),
}
# 값이 '안전 쪽'으로 치우쳤을 때의 표현(위험 방향 반대). 방향 0 신호는 값이 높을 때/낮을 때 표현을 각각 둔다.
SAFE_PHRASE = {
    "grade_level": "직급이 높은 편", "is_leader": "보직자", "is_fixed_term": "정규직", "contract_ending_90d": "계약 만료 여유",
    "base_annual_krw10k": "연봉이 높은 편", "band_position": "보상 밴드 상단", "months_since_raise": "최근 인상 있음", "last_raise_pct": "인상률 높음",
    "promotion_holds_2y": "승진 보류 없음", "months_since_promotion": "최근 승진", "last_rating_score": "최근 평가 높음", "rating_trend": "평가 상승",
    "self_manager_gap": "자기평가와 확정등급 일치", "ot_hours_m6": "연장근로 적음", "ot_trend": "연장근로 감소 추세", "late_count_m6": "지각 없음",
    "absent_days_m6": "결근 없음", "sick_days_m6": "병가 적음", "leave_rejected_12m": "휴가 반려 없음", "training_hours_12m": "교육 참여 많음",
    "oncall_weeks_12m": "온콜 부담 적음", "survey_engagement": "설문 몰입도 높음", "survey_pay_perception": "보상이 적절하다고 응답",
    "survey_workload": "워라밸 응답 높음", "survey_enps": "회사 추천 의향 높음", "survey_nonresponse": "설문 응답함",
}
NEUTRAL_PHRASE = {   # 방향 0 신호: (값 높을 때, 값 낮을 때)
    "tenure_years": ("근속이 긴 편", "근속이 짧은 편(1~3년 구간)"), "age": ("연령대 높은 편", "연령대 낮은 편"),
    "on_leave": ("휴직 중", "재직"), "transfers_count": ("전보 경험 많음", "전보 없음"), "leave_usage_rate": ("연차 사용률 높음", "연차 사용률 낮음"),
}


def phrase_for(c: str, z: float) -> str:
    """표준화 값 z의 부호에 맞는 평이한 표현. 모델 모드처럼 기여 부호가 가설 방향과 다를 때도 값 자체를 바르게 설명한다."""
    d = FEATURE_DOCS.get(c, ("", "", 0))[2]
    if d == 0:
        hi, lo = NEUTRAL_PHRASE.get(c, (FEATURE_DOCS.get(c, (c,))[0] + " 높음", FEATURE_DOCS.get(c, (c,))[0] + " 낮음"))
        return hi if z >= 0 else lo
    risky = (z >= 0) == (d > 0)
    return PLAIN.get(c, (FEATURE_DOCS.get(c, (c,))[0],))[0] if risky else SAFE_PHRASE.get(c, FEATURE_DOCS.get(c, (c,))[0] + " 양호")


def check_question(c: str) -> str:
    return PLAIN.get(c, ("", ""))[1] if len(PLAIN.get(c, ())) > 1 else ""


def _months_between(a, b) -> float:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + (b.month - a.month) + (b.day - a.day) / 30.0


def build_features(ds: Dataset) -> pd.DataFrame:
    """민감 속성은 어떤 옵션으로도 싣지 않는다(정책: config.SENSITIVE_COLUMNS)."""
    if not ds.manifest.get("as_of"):
        raise ValueError("기준일(as_of)이 없습니다. 선검사(precheck)를 먼저 통과해야 합니다.")
    if ds.kind == "roster":
        return _features_from_roster(ds)
    as_of = pd.Timestamp(ds.as_of)
    snap = ds["employees_current"]
    emp = ds["employees"]
    if not len(snap) and len(emp):
        snap = emp.copy()
    if not len(snap):
        raise ValueError("employees_current 또는 employees 시트가 필요합니다.")
    if "employment_status" in snap:
        snap = snap[snap.employment_status.isin(["재직", "휴직"])].copy()
    if "hire_date" in snap:
        snap = snap[pd.to_datetime(snap.hire_date, errors="coerce") <= as_of].copy()
    snap = snap.set_index("employee_id")
    emp = emp.set_index("employee_id") if len(emp) else pd.DataFrame(index=snap.index)
    ids = snap.index.tolist()
    f = pd.DataFrame(index=pd.Index(ids, name="employee_id"))
    unavailable: list[str] = []
    excluded: dict[str, int] = {}
    unparseable: dict[str, int] = {}
    date_checked: list[str] = []
    unchecked: list[str] = []

    def cutoff(df: pd.DataFrame, table: str, date_col: str) -> pd.DataFrame:
        """기준일 이후 기록 제외(누수 차단). 날짜를 읽지 못한 행도 시점을 확인할 수 없으므로 제외하고 건수를 기록한다."""
        if not len(df):
            return df
        if date_col not in df:
            unchecked.append(f"{table}.{date_col}")
            return df.iloc[:0].copy()
        date_checked.append(f"{table}.{date_col}")
        d = pd.to_datetime(df[date_col], errors="coerce")
        after = d.notna() & (d > as_of)
        bad = d.isna()
        excluded[table] = excluded.get(table, 0) + int(after.sum())
        if int(bad.sum()):
            unparseable[table] = unparseable.get(table, 0) + int(bad.sum())
        out = df[~after & ~bad]
        if date_col != "recorded_at" and "recorded_at" in out:
            out = cutoff(out, table + ".recorded", "recorded_at")
        return out

    def col(df, name, default=np.nan):
        return df[name].reindex(ids) if name in df else pd.Series(default, index=ids)

    f["name"] = col(snap, "name", None).fillna(col(emp, "name", None))
    for c in CATEGORICAL_INFO:
        f[c] = col(snap, c, None) if c in snap else col(emp, c, None)

    # ---- 기본 ----
    if "tenure_years" in snap:
        f["tenure_years"] = pd.to_numeric(snap["tenure_years"], errors="coerce")
    elif "hire_date" in emp or "hire_date" in snap:
        hd = pd.to_datetime(col(emp if "hire_date" in emp else snap, "hire_date"), errors="coerce")
        f["tenure_years"] = ((as_of - hd).dt.days / 365.25).round(2)
    else:
        f["tenure_years"] = np.nan
    if "age" in snap:
        f["age"] = pd.to_numeric(snap["age"], errors="coerce")
    elif "birth_date" in emp:
        f["age"] = ((as_of - pd.to_datetime(emp.loc[ids, "birth_date"], errors="coerce")).dt.days // 365.25)
    else:
        f["age"] = np.nan
    f["grade_level"] = pd.to_numeric(col(snap, "grade_level"), errors="coerce")
    f["is_leader"] = (col(snap, "role_title", "팀원").fillna("팀원") != "팀원").astype(int)
    f["is_fixed_term"] = col(snap, "employment_type", "").fillna("").astype(str).str.startswith("기간제").astype(int)
    ce = pd.to_datetime(col(snap, "contract_end_date"), errors="coerce")
    f["contract_ending_90d"] = (ce.notna() & (ce >= as_of) & (ce <= as_of + pd.Timedelta(days=90))).astype(int)
    f["on_leave"] = (col(snap, "employment_status", "재직") == "휴직").astype(int)
    f["base_annual_krw10k"] = pd.to_numeric(col(snap, "base_annual_krw10k"), errors="coerce")

    # ---- 보상 ----
    comp = cutoff(ds["compensation_history"], "compensation_history", "effective_from")
    if len(comp):
        comp = comp[comp.employee_id.isin(ids)].sort_values("effective_from")
        cur = comp.groupby("employee_id").last()
        span = (pd.to_numeric(cur["band_max_krw10k"]) - pd.to_numeric(cur["band_min_krw10k"])).replace(0, np.nan)
        f["band_position"] = ((pd.to_numeric(cur["base_annual_krw10k"]) - pd.to_numeric(cur["band_min_krw10k"])) / span).clip(0, 1).reindex(ids)
        f["months_since_raise"] = cur["effective_from"].map(lambda d: _months_between(d, as_of)).reindex(ids)
        f["last_raise_pct"] = pd.to_numeric(cur["change_pct"], errors="coerce").reindex(ids)
    else:
        f["band_position"] = f["months_since_raise"] = f["last_raise_pct"] = np.nan
        unavailable += ["band_position", "months_since_raise", "last_raise_pct"]

    # ---- 승진·발령 ----
    promo = cutoff(ds["promotion_reviews"], "promotion_reviews", "decided_at")
    if len(promo):
        recent_years = {as_of.year, as_of.year - 1}
        holds = promo[(promo.result == "보류") & pd.to_numeric(promo.review_year, errors="coerce").isin(recent_years)].groupby("employee_id").size()
        f["promotion_holds_2y"] = holds.reindex(ids).fillna(0)
    else:
        f["promotion_holds_2y"] = np.nan
        unavailable.append("promotion_holds_2y")
    acts = cutoff(ds["personnel_actions"], "personnel_actions", "effective_date")
    if len(acts):
        acts = acts[acts.employee_id.isin(ids)]
        last_up = acts[acts.action_type.isin(["승진", "임원 선임", "대표이사 선임", "입사"])].groupby("employee_id").effective_date.max()
        f["months_since_promotion"] = last_up.map(lambda d: _months_between(d, as_of)).reindex(ids)
        f["transfers_count"] = acts[acts.action_type == "전보"].groupby("employee_id").size().reindex(ids).fillna(0)
    else:
        f["months_since_promotion"] = f["transfers_count"] = np.nan
        unavailable += ["months_since_promotion", "transfers_count"]

    # ---- 평가 ----
    rv = cutoff(ds["performance_reviews"], "performance_reviews", "finalized_at")
    f["last_rating_score"] = f["rating_trend"] = f["self_manager_gap"] = np.nan
    if len(rv):
        rv = rv[rv.employee_id.isin(ids) & rv.final_rating.notna()].sort_values("finalized_at").copy()
        rv["score"] = rv.final_rating.map(C.RATING_SCORE)
        if len(rv):
            last = rv.groupby("employee_id").nth(-1).set_index("employee_id")
            prev = rv.groupby("employee_id").nth(-2).set_index("employee_id")
            f["last_rating_score"] = last["score"].reindex(ids)
            if len(prev):
                f["rating_trend"] = (last["score"] - prev["score"].reindex(last.index)).reindex(ids)
            f["self_manager_gap"] = (pd.to_numeric(last["self_score"], errors="coerce") - last["score"]).reindex(ids)
    elif "last_rating" in snap:
        f["last_rating_score"] = snap["last_rating"].map(C.RATING_SCORE).reindex(ids)
        unavailable += ["rating_trend", "self_manager_gap"]
    else:
        unavailable += ["last_rating_score", "rating_trend", "self_manager_gap"]

    # ---- 근태 ----
    am = ds["attendance_monthly"]
    if len(am):
        am = am[am.employee_id.isin(ids)].copy()
        date_checked.append("attendance_monthly.pay_month")
        mdt = pd.to_datetime(am.pay_month.astype(str), format="%Y-%m", errors="coerce")
        bad = mdt.isna()
        if int(bad.sum()):
            unparseable["attendance_monthly"] = int(bad.sum())
        am = am[~bad].copy()
        am["m"] = pd.to_datetime(am.pay_month.astype(str), format="%Y-%m").dt.to_period("M")
        cur_m = pd.Period(as_of, freq="M")
        if not as_of.is_month_end:
            cur_m -= 1
        excluded["attendance_monthly"] = int((am.m > cur_m).sum())
        am = am[am.m <= cur_m]
        for c in ("overtime_hours", "late_count", "absent_days", "sick_days"):
            am[c] = pd.to_numeric(am[c], errors="coerce")
        recent = am[(am.m > cur_m - C.RECENT_MONTHS) & (am.m <= cur_m)]
        g = recent.groupby("employee_id")
        f["ot_hours_m6"] = g.overtime_hours.mean().reindex(ids)
        f["late_count_m6"] = g.late_count.sum().reindex(ids).fillna(0)
        f["absent_days_m6"] = g.absent_days.sum().reindex(ids).fillna(0)
        f["sick_days_m6"] = g.sick_days.sum().reindex(ids).fillna(0)
        last3 = am[(am.m > cur_m - C.TREND_MONTHS) & (am.m <= cur_m)].groupby("employee_id").overtime_hours.mean()
        prior3 = am[(am.m > cur_m - 2 * C.TREND_MONTHS) & (am.m <= cur_m - C.TREND_MONTHS)].groupby("employee_id").overtime_hours.mean()
        f["ot_trend"] = (last3 - prior3).reindex(ids)
    else:
        for c in ("ot_hours_m6", "ot_trend", "late_count_m6", "absent_days_m6", "sick_days_m6"):
            f[c] = np.nan
            unavailable.append(c)
    granted = pd.to_numeric(col(snap, "annual_leave_granted"), errors="coerce").replace(0, np.nan)
    used = pd.to_numeric(col(snap, "annual_leave_used"), errors="coerce")
    f["leave_usage_rate"] = (used / granted).clip(0, 1.5)
    lr = cutoff(ds["leave_requests"], "leave_requests", "requested_at")
    if len(lr):
        lr = lr[lr.employee_id.isin(ids) & (lr.requested_at >= as_of - pd.Timedelta(days=365))]
        f["leave_rejected_12m"] = lr[lr.status == "반려"].groupby("employee_id").size().reindex(ids).fillna(0)
    else:
        f["leave_rejected_12m"] = np.nan
        unavailable.append("leave_rejected_12m")

    # ---- 교육·온콜 ----
    te = cutoff(ds["training_enrollments"], "training_enrollments", "completed_at")
    if len(te):
        te = te[te.employee_id.isin(ids) & (te.status == "수료") & (te.completed_at >= as_of - pd.Timedelta(days=365))]
        f["training_hours_12m"] = pd.to_numeric(te.hours, errors="coerce").groupby(te.employee_id).sum().reindex(ids).fillna(0)
    else:
        f["training_hours_12m"] = np.nan
        unavailable.append("training_hours_12m")
    oc = cutoff(ds["oncall_shifts"], "oncall_shifts", "week_start")
    if len(oc):
        oc = oc[oc.employee_id.isin(ids) & (oc.week_start >= as_of - pd.Timedelta(days=365))]
        f["oncall_weeks_12m"] = oc.groupby("employee_id").size().reindex(ids).fillna(0)
    else:
        f["oncall_weeks_12m"] = 0.0

    # ---- 설문(최근 설문 1개) ----
    for c in ("survey_engagement", "survey_pay_perception", "survey_workload", "survey_enps"):
        f[c] = np.nan
    f["survey_nonresponse"] = np.nan
    sv, sq, sr, sa = cutoff(ds["surveys"], "surveys", "closed_at"), ds["survey_questions"], ds["survey_responses"], ds["survey_answers"]
    if len(sv) and len(sq) and len(sa):
        latest = sv.sort_values("closed_at").survey_id.iloc[-1]
        q = sq[sq.survey_id == latest].set_index("question_id")
        ans = sa[sa.question_id.isin(q.index) & sa.employee_id.isin(ids)].copy()
        ans["dimension"] = ans.question_id.map(q.dimension)
        ans["value"] = pd.to_numeric(ans.value, errors="coerce")
        piv = ans.pivot_table(index="employee_id", columns="dimension", values="value", aggfunc="mean")
        eng_cols = [c for c in ("몰입", "성장", "관리자", "자율성") if c in piv]
        if eng_cols:
            f["survey_engagement"] = piv[eng_cols].mean(axis=1).reindex(ids)
        for dim, c in (("보상 인식", "survey_pay_perception"), ("워라밸", "survey_workload"), ("추천의향", "survey_enps")):
            if dim in piv:
                f[c] = piv[dim].reindex(ids)
        if len(sr):
            resp = sr[sr.survey_id == latest].set_index("employee_id")
            f["survey_nonresponse"] = resp.status.reindex(ids).eq("미응답").astype(int)
    else:
        unavailable += ["survey_engagement", "survey_pay_perception", "survey_workload", "survey_enps", "survey_nonresponse"]

    f.attrs["unavailable"] = sorted(set(unavailable))
    f.attrs["excluded_after_as_of"] = excluded
    f.attrs["unparseable_dates"] = unparseable
    f.attrs["date_columns_checked"] = date_checked
    f.attrs["unchecked_date_tables"] = unchecked
    f.attrs["has_history"] = True
    return f.reset_index()


def _features_from_roster(ds: Dataset) -> pd.DataFrame:
    """직원 명부 한 장: 있는 컬럼으로만 신호를 만든다."""
    r = ds["roster"].set_index("employee_id")
    as_of = pd.Timestamp(ds.as_of)
    if "employment_status" in r:
        r = r[r.employment_status.isin(["재직", "휴직"])].copy()
    if "hire_date" in r:
        r = r[pd.to_datetime(r.hire_date, errors="coerce") <= as_of].copy()
    ids = r.index.tolist()
    f = pd.DataFrame(index=pd.Index(ids, name="employee_id"))
    f["name"] = r["name"] if "name" in r else ids
    for c in CATEGORICAL_INFO:
        f[c] = r[c] if c in r else None
    for c in ("division_name", "org_unit_name", "job_family", "grade_title", "role_title"):
        if f[c].isna().all():
            f[c] = {"division_name": "전체", "org_unit_name": "미지정", "job_family": "미지정", "grade_title": "미지정", "role_title": "팀원"}[c]
    for c in NUMERIC_FEATURES:
        f[c] = np.nan
    if "hire_date" in r:
        f["tenure_years"] = ((as_of - pd.to_datetime(r["hire_date"], errors="coerce")).dt.days / 365.25).round(2)
    if "age" in r:
        f["age"] = pd.to_numeric(r["age"], errors="coerce")
    elif "birth_date" in r:
        f["age"] = ((as_of - pd.to_datetime(r["birth_date"], errors="coerce")).dt.days // 365.25)
    if "grade_level" in r:
        f["grade_level"] = pd.to_numeric(r["grade_level"], errors="coerce")
    f["is_leader"] = (f["role_title"].fillna("팀원") != "팀원").astype(int)
    f["is_fixed_term"] = f["employment_type"].fillna("").astype(str).str.contains("기간제|계약").astype(int)
    if "contract_end_date" in r:
        ce = pd.to_datetime(r["contract_end_date"], errors="coerce")
        f["contract_ending_90d"] = (ce.notna() & (ce >= as_of) & (ce <= as_of + pd.Timedelta(days=90))).astype(int)
    else:
        f["contract_ending_90d"] = 0
    f["on_leave"] = f["employment_status"].fillna("").astype(str).str.contains("휴직").astype(int)
    if "base_annual_krw10k" in r:
        f["base_annual_krw10k"] = pd.to_numeric(r["base_annual_krw10k"], errors="coerce")
    if "last_raise_date" in r:
        f["months_since_raise"] = pd.to_datetime(r["last_raise_date"], errors="coerce").map(lambda d: _months_between(d, as_of) if pd.notna(d) else np.nan)
    if "last_raise_pct" in r:
        f["last_raise_pct"] = pd.to_numeric(r["last_raise_pct"], errors="coerce")
    if "last_promotion_date" in r:
        f["months_since_promotion"] = pd.to_datetime(r["last_promotion_date"], errors="coerce").map(lambda d: _months_between(d, as_of) if pd.notna(d) else np.nan)
    if "promotion_holds" in r:
        f["promotion_holds_2y"] = pd.to_numeric(r["promotion_holds"], errors="coerce")
    if "last_rating" in r:
        f["last_rating_score"] = r["last_rating"].map(lambda v: C.RATING_SCORE.get(str(v).strip().upper(), np.nan) if pd.notna(v) else np.nan)
        if "prev_rating" in r:
            prev = r["prev_rating"].map(lambda v: C.RATING_SCORE.get(str(v).strip().upper(), np.nan) if pd.notna(v) else np.nan)
            f["rating_trend"] = f["last_rating_score"] - prev
    if "overtime_hours_monthly" in r:
        f["ot_hours_m6"] = pd.to_numeric(r["overtime_hours_monthly"], errors="coerce")
    for src, dst in (("late_count", "late_count_m6"), ("absent_days", "absent_days_m6"), ("sick_days", "sick_days_m6"),
                     ("training_hours", "training_hours_12m"), ("survey_engagement", "survey_engagement"), ("survey_enps", "survey_enps")):
        if src in r:
            f[dst] = pd.to_numeric(r[src], errors="coerce")
    if "annual_leave_granted" in r and "annual_leave_used" in r:
        g = pd.to_numeric(r["annual_leave_granted"], errors="coerce").replace(0, np.nan)
        f["leave_usage_rate"] = (pd.to_numeric(r["annual_leave_used"], errors="coerce") / g).clip(0, 1.5)
    # 명부 모드: 날짜 컬럼이 있으면 기준일 이후 값(미래 입사일 등)을 결측 처리하고 건수를 기록한다
    excluded, unparseable, checked = {}, {}, []
    for c in ("hire_date", "last_raise_date", "last_promotion_date"):
        if c in r:
            checked.append(f"roster.{c}")
            d = pd.to_datetime(r[c], errors="coerce")
            bad = d.isna() & r[c].notna()
            bad_count = max(int(bad.sum()), int(r.attrs.get("date_parse_errors", {}).get(c, 0)))
            if bad_count:
                unparseable[f"roster.{c}"] = bad_count
            if int((d > as_of).sum()):
                excluded[f"roster.{c}"] = int((d > as_of).sum())
            feature = {"hire_date": "tenure_years", "last_raise_date": "months_since_raise", "last_promotion_date": "months_since_promotion"}[c]
            f.loc[d.isna() | (d > as_of), feature] = np.nan
    if "hire_date" in r:
        hd = pd.to_datetime(r["hire_date"], errors="coerce")
        f.loc[(hd > as_of).reindex(f.index).fillna(False).to_numpy(), "tenure_years"] = np.nan
    f.attrs["unavailable"] = [c for c in NUMERIC_FEATURES if f[c].isna().all()]
    f.attrs["excluded_after_as_of"] = excluded
    f.attrs["unparseable_dates"] = unparseable
    f.attrs["date_columns_checked"] = checked
    f.attrs["has_history"] = False
    return f.reset_index()


ALLOWED_FEATURES = tuple(k for k in FEATURE_DOCS if k not in C.EXCLUDED_FROM_INDIVIDUAL_WEIGHTS)


def numeric_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """점수 계산용 행렬. 허용 신호(ALLOWED_FEATURES)만 들어가며 민감 속성은 이름이 섞여 있어도 걸러낸다."""
    leaked = [c for c in features.columns if c in C.SENSITIVE_COLUMNS]
    if leaked:
        raise ValueError(f"민감 속성이 피처 테이블에 들어 있습니다: {leaked}")
    cols = [c for c in ALLOWED_FEATURES if c in features]
    return features.set_index("employee_id")[cols].astype(float)


def hr_value(feature, value, reference=None):
    """Observed HR value, with an explicit comparison reference; no causal claim."""
    if pd.isna(value):
        return f'{FEATURE_DOCS.get(feature, (feature,))[0]}: 데이터 없음'
    label = FEATURE_DOCS.get(feature, (feature,))[0]
    units = {'tenure_years':'년', 'base_annual_krw10k':'만원', 'months_since_raise':'개월',
             'months_since_promotion':'개월', 'last_raise_pct':'%', 'ot_hours_m6':'시간/월',
             'ot_trend':'시간/월', 'late_count_m6':'회', 'absent_days_m6':'일',
             'promotion_holds_2y':'회', 'leave_rejected_12m':'회', 'training_hours_12m':'시간',
             'oncall_weeks_12m':'주', 'survey_enps':'점/10', 'survey_engagement':'점/5',
             'survey_pay_perception':'점/5', 'survey_workload':'점/5', 'last_rating_score':'점/5'}
    if feature in ('is_leader','is_fixed_term','contract_ending_90d','on_leave'):
        return label + (': 해당' if value else ': 해당 없음')
    if feature == 'band_position':
        return f'보상 밴드 내 위치 {value*100:.0f}%'
    if feature == 'leave_usage_rate':
        return f'연차 사용률 {value*100:.0f}%'
    if feature == 'ot_trend':
        return f'최근 3개월 월평균 연장근로 {abs(value):.1f}시간 {"증가" if value>=0 else "감소"}(직전 3개월 대비)'
    unit = units.get(feature,'')
    text = f'{label} {value:.1f}{unit}'
    if reference is not None and pd.notna(reference):
        text += f' (비교 기준 {reference:.1f}{unit})'
    return text
