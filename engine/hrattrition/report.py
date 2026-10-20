"""HR 담당자용 요약(summary.json)과 결과 워크북(퇴사위험_결과.xlsx).

결과 워크북은 사람이 읽는 한글 시트들과, 대시보드 스킬이 읽는 기계용 시트(_data, _meta)를 함께 담는다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURE_DOCS, PLAIN, hr_value
from .model import ScoreResult

ACTIONS = {   # 신호 → (조치 범주, 구체 제안)
    "band_position": ("보상", "밴드 하단 직원의 보상 위치를 점검하고 다음 조정 시점을 계획"),
    "months_since_raise": ("보상", "장기간 인상 없는 직원의 조정 이력을 확인"),
    "last_raise_pct": ("보상", "인상 폭에 대한 인식을 1:1에서 확인"),
    "survey_pay_perception": ("보상", "보상 인식이 낮은 그룹과 실제 밴드 위치를 비교"),
    "promotion_holds_2y": ("성장", "승진 보류 사유를 본인에게 설명하고 다음 심사 로드맵을 제시"),
    "months_since_promotion": ("성장", "경력 경로·역할 확장 기회를 논의"),
    "training_hours_12m": ("성장", "교육·학습 기회를 배정"),
    "last_rating_score": ("성과", "낮은 평가의 원인과 개선 지원을 정리"),
    "rating_trend": ("성과", "평가 하락 직원의 업무·관계 변화를 확인"),
    "self_manager_gap": ("성과", "평가 결과 소통(캘리브레이션 설명)을 강화"),
    "ot_hours_m6": ("업무량", "연장근로 상위 그룹의 인력 배치·업무 분배를 조정"),
    "ot_trend": ("업무량", "최근 급증한 연장근로의 원인(마감·결원)을 확인"),
    "oncall_weeks_12m": ("업무량", "온콜 순번을 재분배"),
    "survey_workload": ("업무량", "워라밸 응답이 낮은 팀의 근무시간을 점검"),
    "late_count_m6": ("근태", "지각·결근 패턴 변화는 건강·동기 신호일 수 있어 면담"),
    "absent_days_m6": ("근태", "결근 사유를 파악"),
    "sick_days_m6": ("근태", "건강 상태·스트레스 요인을 확인"),
    "leave_rejected_12m": ("근태", "휴가 승인 관행·대체 인력을 점검"),
    "survey_engagement": ("몰입", "몰입도가 낮은 팀의 관리자 1:1·업무 의미 부여를 강화"),
    "survey_enps": ("몰입", "추천 의향이 낮은 그룹의 피드백을 수집"),
    "survey_nonresponse": ("몰입", "설문 미응답자와 소통 채널을 확인"),
    "is_fixed_term": ("계약", "기간제 직원의 전환 계획을 공유"),
    "contract_ending_90d": ("계약", "계약 만료 90일 전 연장·전환 의사를 확인"),
    "tenure_years": ("초기 적응", "근속 1~3년 구간의 온보딩 이후 경력 계획을 점검"),
}
KO = {"employee_id": "사번", "name": "이름", "division_name": "본부", "org_unit_name": "팀", "job_family": "직군",
      "grade_title": "직급", "role_title": "직책", "employment_type": "고용형태", "employment_status": "재직상태",
      "tenure_years": "근속(년)", "risk_score": "위험지수", "risk_band": "위험밴드", "drivers_plain": "주요 신호",
      "top_drivers": "요인(기술)"}


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    return v


def build_summary(features: pd.DataFrame, res: ScoreResult, manifest: dict, top_n: int = 10, question: dict | None = None,
                  data_profile: dict | None = None, hypotheses: dict | None = None, validation: dict | None = None,
                  population: dict | None = None) -> dict:
    df = features.merge(res.scores, on="employee_id")
    for c in ("division_name", "org_unit_name"):
        df[c] = df[c].fillna("미지정")
    n = len(df)
    bands = {b: int((df.risk_band == b).sum()) for b in ("높음", "주의", "낮음")}
    by_unit = (df.groupby(["division_name", "org_unit_name"]).agg(avg_risk=("risk_score", "mean"), n=("employee_id", "size"),
                                                                 high=("risk_band", lambda s: int((s == "높음").sum())))
               .reset_index().sort_values("avg_risk", ascending=False))
    by_div = df.groupby("division_name").agg(avg_risk=("risk_score", "mean"), n=("employee_id", "size")).reset_index().sort_values("avg_risk", ascending=False)
    top = df.sort_values("risk_score", ascending=False).head(top_n)
    top_cols = ["employee_id", "name", "division_name", "org_unit_name", "grade_title", "tenure_years", "risk_score", "risk_band", "drivers_plain"]

    signals, actions = [], []
    if res.contrib is not None and (df.risk_band == "높음").any():
        high_ids = df.loc[df.risk_band == "높음", "employee_id"]
        mean_contrib = res.contrib.reindex(high_ids).mean().sort_values(ascending=False)
        for feat, v in mean_contrib.head(5).items():
            if v <= (0.005 if res.mode == "model" else 0.1):
                break
            observed_mean = df.loc[df.employee_id.isin(high_ids), feat].mean()
            label = (hr_value(feat, observed_mean), "원시 값과 최근 변경 이력을 확인")
            signals.append(dict(feature=feat, plain=label[0], check=label[1] if len(label) > 1 else "", avg_contribution=round(float(v), 2)))
            if feat in ACTIONS:
                cat, act = ACTIONS[feat]
                if res.mode == "model":
                    act = "실제 값·최근 변동·직원 상황을 확인한 뒤 지원 조치 여부를 결정"
                actions.append(dict(category=cat, action=act, signal=label[0]))
    unavailable = features.attrs.get("unavailable", [])
    # 5단계 행동 연결: 상위 팀별로 '확인할 사항'을 붙인다
    unit_actions = []
    if res.contrib is not None:
        for _, u in by_unit.head(5).iterrows():
            ids_u = df.loc[(df.division_name == u.division_name) & (df.org_unit_name == u.org_unit_name), "employee_id"]
            mc = res.contrib.reindex(ids_u).mean().sort_values(ascending=False)
            checks = ["최근 변경 이력과 해당 직원의 상황을 확인" for c in mc.head(2).index if c in PLAIN and mc[c] > 0.005]
            unit_actions.append(dict(division_name=u.division_name, org_unit_name=u.org_unit_name, n=int(u.n), avg_risk=round(float(u.avg_risk), 3),
                                     signals=[hr_value(c, df.loc[df.employee_id.isin(ids_u), c].mean()) for c in mc.head(2).index if c in PLAIN and mc[c] > 0.005], hr_checks=checks))
    analysis_mode = (question or {}).get("analysis_mode", "diagnostic")
    mode_label = {"predictive": "예측 모드(라벨 기반 모델)", "change": "변화 비교 모드", "diagnostic": "진단 모드(규칙 기반 신호 지수 · 예측 아님)"}[analysis_mode]
    return dict(
        question=question, population=population, data_profile=data_profile, hypotheses=hypotheses, validation=validation,
        unit_actions=unit_actions, analysis_mode=analysis_mode, calc_mode=res.mode,
        company=manifest.get("company", {}).get("name"), as_of=manifest.get("as_of"), as_of_source=manifest.get("as_of_source"),
        source_file=manifest.get("source_file"),
        n_employees=n, mode=res.mode, mode_label=mode_label,
        model_metrics={k: res.metrics[k] for k in ("cv_auc", "positives", "n_train") if k in res.metrics},
        fallback_reason=res.metrics.get("fallback_reason"),
        bands=bands, bands_pct={b: round(100 * c / max(n, 1), 1) for b, c in bands.items()},
        avg_risk=round(float(df.risk_score.mean()), 3),
        by_division=[{k: _clean(v) for k, v in r.items()} for r in by_div.head(10).to_dict("records")],
        top_units=[{k: _clean(v) for k, v in r.items()} for r in by_unit.head(5).to_dict("records")],
        top_employees=[{k: _clean(v) for k, v in r.items()} for r in top[top_cols].to_dict("records")],
        company_signals=signals, actions=actions,
        unavailable_signals=[FEATURE_DOCS[c][0] for c in unavailable if c in FEATURE_DOCS],
        caveats=[
            "진단 모드의 지수는 예측 확률이 아니라 여러 신호를 모은 회사 내 상대 순위다. 개인 평가나 인사 조치의 단독 근거로 쓰면 안 된다."
            if res.mode != "model" else "고정 모델의 점수는 확인 우선순위용이며 보정된 퇴사 확률이 아니다. 성능은 별도 미래 기간 평가를 참고한다.",
            "성별·국적·체류자격·혼인·가족·주거·거주지·병역은 계산에 쓰지 않았다. 병가·설문 미응답은 개인 점수에서 제외했다.",
            "신호는 '확인해 볼 지점'이며 직원 개인의 성향이나 태도를 뜻하지 않는다.",
        ],
    )


def write_summary(summary: dict, out_dir: Path) -> Path:
    p = out_dir / "summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def write_result_workbook(features: pd.DataFrame, res: ScoreResult, summary: dict, manifest: dict, out_dir: Path) -> Path:
    df = features.merge(res.scores, on="employee_id").sort_values("risk_score", ascending=False)
    staff = df[[c for c in KO if c in df]].rename(columns=KO)
    feat_cols = ["employee_id", "name"] + [c for c in FEATURE_DOCS if c in df]
    feats = df[feat_cols].rename(columns={"employee_id": "사번", "name": "이름", **{c: FEATURE_DOCS[c][0] for c in FEATURE_DOCS}})
    overview = pd.DataFrame([
        ("회사", summary["company"]), ("기준일", summary["as_of"]), ("원본 파일", summary.get("source_file")),
        ("분석 인원", summary["n_employees"]), ("모드", summary["mode_label"]),
        ("위험 높음", f"{summary['bands']['높음']}명 ({summary['bands_pct']['높음']}%)"),
        ("주의", f"{summary['bands']['주의']}명 ({summary['bands_pct']['주의']}%)"),
        ("낮음", f"{summary['bands']['낮음']}명 ({summary['bands_pct']['낮음']}%)"),
        ("계산하지 못한 신호", ", ".join(summary["unavailable_signals"]) or "없음"),
    ] + [("주의사항", c) for c in summary["caveats"]], columns=["항목", "값"])
    signals = pd.DataFrame(summary["company_signals"]).rename(columns={"feature": "신호코드", "plain": "신호", "check": "확인할 점", "avg_contribution": "평균 기여"}) if summary["company_signals"] else pd.DataFrame(columns=["신호", "확인할 점"])
    actions = pd.DataFrame(summary["actions"]).rename(columns={"category": "범주", "action": "조치 제안", "signal": "근거 신호"}) if summary["actions"] else pd.DataFrame(columns=["범주", "조치 제안"])
    glossary = pd.DataFrame([(FEATURE_DOCS[c][0], FEATURE_DOCS[c][1], {1: "클수록 위험", -1: "클수록 안전", 0: "참고"}[FEATURE_DOCS[c][2]],
                              PLAIN.get(c, ("", ""))[0], PLAIN.get(c, ("", ""))[1]) for c in FEATURE_DOCS],
                            columns=["신호", "계산 방식", "방향", "위험 쪽일 때 표현", "확인 질문"])
    # 기계용 시트: 대시보드 스킬이 그대로 읽는다
    from . import __version__
    meta_rows = [("company", summary["company"]), ("as_of", summary["as_of"]), ("as_of_source", summary.get("as_of_source") or ""),
                 ("source_file", summary.get("source_file") or ""),
                 ("mode", res.mode), ("analysis_mode", summary.get("analysis_mode", "diagnostic")), ("engine_version", __version__),
                 ("metrics_json", json.dumps(res.metrics, ensure_ascii=False)),
                 ("unavailable_json", json.dumps(features.attrs.get("unavailable", []), ensure_ascii=False)),
                 ("manifest_json", json.dumps({k: v for k, v in manifest.items() if k != "column_mapping"}, ensure_ascii=False, default=str)),
                 ("schema", C.RESULT_SCHEMA)]
    # Excel text cells are limited to 32767 characters; preserve large evaluation reports.
    expanded = []
    for key, value in meta_rows:
        if isinstance(value, str) and len(value) > 30000:
            expanded.append((key, "@chunks"))
            expanded.extend((f"{key}_{i//16000:04d}", value[i:i+16000]) for i in range(0, len(value), 16000))
        else:
            expanded.append((key, value))
    meta_rows = expanded
    # 방법론 단계별 시트
    q = summary.get("question") or {}
    question_df = pd.DataFrame([("분석 모드", summary.get("mode_label", "")),
                                ("요청한 질문", q.get("requested_question", "")), ("지금 데이터로 답하는 질문", q.get("answerable_question", "")),
                                ("관심 사건", q.get("target", "")), ("요청 기간", f"기준일 이후 {q.get('horizon_days', '')}일"),
                                ("기준 모집단", q.get("population", "")), ("모집단 제한", json.dumps(q.get("filters", {}), ensure_ascii=False)),
                                ("기준일", f"{q.get('as_of', '')} ({summary.get('as_of_source') or '출처 미확인'})"), ("사용자 요청 원문", q.get("raw_prompt", "")),
                                ("모집단 인원", f"{(summary.get('population') or {}).get('after', summary['n_employees'])}명 / 전체 {(summary.get('population') or {}).get('before', summary['n_employees'])}명")],
                               columns=["항목", "값"])
    prof = summary.get("data_profile") or {}
    profile_df = pd.DataFrame(prof.get("tables", [])).rename(columns={"table": "테이블", "meaning": "의미", "present": "있음", "rows": "행 수", "employees_covered": "직원 커버",
                                                                        "coverage_pct": "커버리지(%)", "unknown_employee_rows": "명단 밖 사번 행", "duplicate_ids": "중복 사번", "date_span": "기간", "months": "월 수"})
    issues_df = pd.DataFrame([("이슈", i) for i in prof.get("issues", [])] + [("신호 결측", f"{m['signal']} {m['missing_pct']}%") for m in prof.get("missing_signals", [])], columns=["구분", "내용"])
    hyp = summary.get("hypotheses") or {}
    hyp_rows = []
    for h in hyp.get("hypotheses", []):
        if not h.get("signals"):
            hyp_rows.append(dict(가설ID=h["id"], 가설=h["hypothesis"], 판정=h["status"], 신호="", 비고=h.get("note", "")))
        for s in h.get("signals", []):
            hyp_rows.append(dict(가설ID=h["id"], 가설=h["hypothesis"], 판정=h["status"], 신호=s["signal"], 두드러진인원=s.get("strong_n"), 비율=s.get("strong_pct"),
                                 집중팀=s.get("concentrated_in", ""), 높음그룹동반비율=s.get("share_in_high_pct"), AUC=s.get("auc"), 비고=h.get("note", "")))
    hyp_df = pd.DataFrame(hyp_rows)
    val = summary.get("validation") or {}
    val_rows = [("종합", f"{val.get('overall_ko', '')}")]
    val_rows += [("체크리스트", f"[{c['id']}] {c['item']}: {c.get('status_ko', c.get('status'))} — {c['detail']}" + (f" (근거: {', '.join(c['evidence'][:3])})" if c.get("evidence") else ""))
                 for c in val.get("checklist", [])]
    val_rows += [("기준일 이후 제외", f"{k}: {v}행") for k, v in (val.get("leakage") or {}).get("excluded_rows_by_table", {}).items()]
    val_rows += [("날짜 해석 실패 제외", f"{k}: {v}행") for k, v in (val.get("leakage") or {}).get("unparseable_rows_by_table", {}).items()]
    val_rows += [("반례", f"[{c['hypothesis']}] {c['type']} {c['count']}건 (예: {', '.join(c['examples'])}) — {c['reading']}") for c in val.get("counterexamples", [])]
    if val.get("sensitivity"):
        val_rows += [("순위 안정성", f"{r['signal']} 제거 시 순위 상관 {r['rank_corr_without']}, 밴드 변경 {r['band_changes']}명") for r in val["sensitivity"]["most_influential"]]
    val_rows += [("한계", l) for l in val.get("limitations", [])]
    val_df = pd.DataFrame(val_rows, columns=["구분", "내용"])
    unit_act = pd.DataFrame(summary.get("unit_actions", []))
    if len(unit_act):
        unit_act = unit_act.assign(signals=unit_act.signals.map(lambda x: "; ".join(x)), hr_checks=unit_act.hr_checks.map(lambda x: "; ".join(x))) \
            .rename(columns={"division_name": "본부", "org_unit_name": "팀", "n": "인원", "avg_risk": "평균 위험", "signals": "두드러진 신호", "hr_checks": "HR이 확인할 사항"})

    p = out_dir / C.RESULT_FILENAME
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        question_df.to_excel(w, sheet_name="1 질문 정의", index=False)
        overview.to_excel(w, sheet_name="요약", index=False)
        profile_df.to_excel(w, sheet_name="2 데이터 이해", index=False)
        issues_df.to_excel(w, sheet_name="2 데이터 이슈", index=False)
        hyp_df.to_excel(w, sheet_name="3 가설", index=False)
        val_df.to_excel(w, sheet_name="4 검증", index=False)
        unit_act.to_excel(w, sheet_name="5 팀별 확인사항", index=False)
        staff.to_excel(w, sheet_name="직원별 위험", index=False)
        df.groupby(["division_name", "org_unit_name"]).agg(avg_risk=("risk_score", "mean"), n=("employee_id", "size"), high=("risk_band", lambda x: int((x == "높음").sum()))).reset_index().rename(columns={"division_name": "본부", "org_unit_name": "팀", "avg_risk": "평균 위험", "n": "인원", "high": "높음 인원"}).to_excel(w, sheet_name="팀별", index=False)
        signals.to_excel(w, sheet_name="회사 전체 신호", index=False)
        actions.to_excel(w, sheet_name="조치 제안", index=False)
        feats.to_excel(w, sheet_name="신호 상세", index=False)
        glossary.to_excel(w, sheet_name="신호 설명", index=False)
        df.to_excel(w, sheet_name=C.DATA_SHEET, index=False)
        pd.DataFrame(meta_rows, columns=["key", "value"]).to_excel(w, sheet_name=C.META_SHEET, index=False)
    return p


def write_analysis_spec(summary: dict, res: ScoreResult, weights_override: dict | None, mode_requested: str, out_dir: Path) -> Path:
    """6단계 재사용(P0-4): 계산 설정만 담는다(입력 해시·실행 시각은 run_manifest.json). 같은 입력·스펙·엔진이면 점수·밴드가 같아야 한다."""
    from . import __version__
    spec = dict(schema=C.SPEC_SCHEMA, engine_version=__version__,
                question={k: v for k, v in (summary.get("question") or {}).items() if k not in ("as_of", "as_of_source")},
                mode_requested=mode_requested, mode_applied=res.mode, model_reference=res.metrics.get("model_reference"), analysis_mode=summary.get("analysis_mode", "diagnostic"),
                weights_override=weights_override or {}, weights_full=res.metrics.get("weights_full") or res.metrics.get("weights"),
                tenure_bump=res.metrics.get("tenure_bump"), scale=res.metrics.get("scale"), bands=res.metrics.get("bands"),
                feature_policy=dict(allowed_features=res.metrics.get("allowed_features"), excluded_from_individual_weights=res.metrics.get("excluded_from_individual_weights"),
                                    sensitive_never_used=res.metrics.get("sensitive_never_used")),
                input_conditions=dict(file="HR xlsx 1개(표준 워크북 또는 직원 명부 한 장)", label_sheet="퇴사라벨/labels (선택)",
                                      as_of="파일의 명시적 기준일 또는 --as-of (오늘 날짜 자동 적용 없음)"),
                procedure=["질문 정의", "데이터 이해(선검사 포함)", "가설 수립·탐색", "검증(누수·반례·산정방식·안정성·커버리지)", "행동 연결", "재사용 스펙 저장"],
                validation_criteria=[f"{c['id']} {c['item']}" for c in (summary.get("validation") or {}).get("checklist", [])],
                output_format=dict(workbook=C.RESULT_FILENAME, result_schema=C.RESULT_SCHEMA, summary="summary.json", run_manifest="run_manifest.json",
                                   dashboard_input="퇴사위험_결과.xlsx(_data/_meta)"))
    p = out_dir / "analysis_spec.json"
    p.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_spec(path: Path) -> dict:
    """스펙 로드와 버전 검사. 지원하지 않는 스키마는 오류(조용히 다른 설정으로 돌지 않는다)."""
    from . import __version__
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = spec.get("schema", "")
    if schema == "hr-attrition-spec/1":
        spec = dict(schema=C.SPEC_SCHEMA, engine_version=spec.get("engine_version"), question=spec.get("question"), mode_requested=spec.get("mode", "auto"),
                    weights_override=spec.get("weights_override") or {}, weights_full=None, migrated_from="hr-attrition-spec/1")
    elif schema != C.SPEC_SCHEMA:
        raise ValueError(f"지원하지 않는 스펙 스키마: {schema!r} (지원: {C.SPEC_SCHEMA})")
    ev = str(spec.get("engine_version") or "")
    if ev and ev.split(".")[:2] != __version__.split(".")[:2]:
        raise ValueError(f"스펙 엔진 버전 {ev}은 현재 {__version__}과 호환되지 않습니다. 같은 minor 버전에서 만든 스펙만 재사용할 수 있습니다.")
    return spec


def write_run_manifest(out_dir: Path, inputs: list[Path], spec_path: Path, warnings: list[str], extra: dict | None = None) -> Path:
    """실행별 값(입력 해시·시각·엔진)과 계산 설정(스펙)을 분리해 기록한다. 해시는 익명화 수단이 아니다."""
    import hashlib
    import uuid
    from datetime import datetime
    from . import __version__

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    man = dict(run_id=uuid.uuid4().hex[:12], executed_at=datetime.now().astimezone().isoformat(timespec="seconds"), engine_version=__version__,
               input_hashes={str(p.name): sha(p) for p in inputs if Path(p).is_file()}, spec_hash=sha(spec_path), warnings=warnings, **(extra or {}))
    p = out_dir / "run_manifest.json"
    p.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_result_extras(path: Path) -> dict:
    """결과 워크북의 사람용 시트에서 대시보드가 쓰는 부가 정보(검증 체크리스트·가설·팀별 확인사항·회사 신호·조치)를 읽는다."""
    import re
    xf = pd.ExcelFile(Path(path), engine="openpyxl")
    out = dict(checklist=[], hypotheses=[], unit_actions=[], signals=[], actions=[], question={})

    def sheet(name):
        return pd.read_excel(xf, sheet_name=name, dtype=str).fillna("") if name in xf.sheet_names else None

    v = sheet("4 검증")
    if v is not None:
        for _, r in v.iterrows():
            if r.get("구분") == "체크리스트":
                m = re.match(r"\[(V\d+)\]\s*(.+?):\s*(통과|경고|실패|평가 불가|해당 없음)\s*—\s*(.*)", str(r.get("내용", "")))
                if m:
                    out["checklist"].append(dict(id=m.group(1), item=m.group(2), status_ko=m.group(3), detail=m.group(4)))
            elif r.get("구분") == "종합":
                out["overall_ko"] = r.get("내용", "")
    h = sheet("3 가설")
    if h is not None and len(h):
        for hid, g in h.groupby("가설ID", sort=True):
            first = g.iloc[0]
            sig = [dict(signal=s["신호"], strong_pct=s.get("비율", ""), concentrated_in=s.get("집중팀", ""), auc=s.get("AUC", "")) for _, s in g.iterrows() if s.get("신호")]
            out["hypotheses"].append(dict(id=hid, hypothesis=first["가설"], status=first["판정"], signals=sig))
    u = sheet("5 팀별 확인사항")
    if u is not None and len(u):
        out["unit_actions"] = [dict(division=r.get("본부", ""), unit=r.get("팀", ""), n=r.get("인원", ""), avg_risk=r.get("평균 위험", ""),
                                    signals=r.get("두드러진 신호", ""), checks=r.get("HR이 확인할 사항", "")) for _, r in u.iterrows()]
    s = sheet("회사 전체 신호")
    if s is not None and len(s) and "신호" in s:
        out["signals"] = [dict(signal=r.get("신호", ""), check=r.get("확인할 점", ""), contribution=r.get("평균 기여", "")) for _, r in s.iterrows()]
    a = sheet("조치 제안")
    if a is not None and len(a) and "조치 제안" in a:
        out["actions"] = [dict(category=r.get("범주", ""), action=r.get("조치 제안", ""), signal=r.get("근거 신호", "")) for _, r in a.iterrows()]
    q = sheet("1 질문 정의")
    if q is not None and len(q):
        out["question"] = dict(zip(q["항목"], q["값"]))
    return out


def read_result_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """예측 결과 워크북 → (features, scores, meta). _data/_meta 시트가 있어야 한다."""
    path = Path(path)
    xf = pd.ExcelFile(path, engine="openpyxl")
    if C.DATA_SHEET not in xf.sheet_names or C.META_SHEET not in xf.sheet_names:
        raise ValueError(f"{path.name}은(는) /predict_attrition이 만든 결과 파일이 아닙니다({C.DATA_SHEET}/{C.META_SHEET} 시트 없음).")
    data = pd.read_excel(xf, sheet_name=C.DATA_SHEET, dtype={"employee_id": str})
    meta_df = pd.read_excel(xf, sheet_name=C.META_SHEET, dtype=str).fillna("")
    meta = dict(zip(meta_df["key"], meta_df["value"]))
    for key, value in list(meta.items()):
        if value == "@chunks":
            meta[key] = "".join(meta[k] for k in sorted(meta) if k.startswith(key + "_") and k[len(key)+1:].isdigit())
    meta["metrics"] = json.loads(meta.get("metrics_json") or "{}")
    meta["unavailable"] = json.loads(meta.get("unavailable_json") or "[]")
    meta["manifest"] = json.loads(meta.get("manifest_json") or "{}")
    if not meta["manifest"].get("company"):
        meta["manifest"]["company"] = {"name": meta.get("company") or path.stem}
    meta["manifest"].setdefault("as_of", meta.get("as_of"))
    score_cols = ["employee_id", "risk_score", "risk_band", "top_drivers", "drivers_plain", "mode"]
    scores = data[[c for c in score_cols if c in data]].copy()
    features = data.drop(columns=[c for c in score_cols if c in data and c != "employee_id"])
    features.attrs["unavailable"] = meta["unavailable"]
    return features, scores, meta
