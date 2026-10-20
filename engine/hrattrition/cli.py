"""명령줄.

    predict   --input <HR.xlsx|폴더|프리셋> --out DIR [--as-of YYYY-MM-DD] [--labels csv] [--mode auto|model|heuristic]
              [--weights w.json] [--spec analysis_spec.json] [--target …] [--horizon-days N] [--filter 키=값]... [--prompt "…"] [--top 10]
              → features.csv, scores.csv, metrics.json, summary.json, 퇴사위험_결과.xlsx, analysis_spec.json, run_manifest.json
              입력 선검사에서 차단 이슈가 있으면 diagnostics.json만 쓰고 종료코드 2.
    dashboard --result <퇴사위험_결과.xlsx> [--data <HR.xlsx>] --out DIR [--open]
    run       predict + dashboard

설정 우선순위: 명시적 CLI > --spec > 기본값. CLI 인자의 기본값은 None이라 스펙을 덮지 않는다.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import config as C
from .dashboard import build_dashboard
from .data import load_any
from .features import build_features
from .hypotheses import evaluate_hypotheses
from .labels import describe_labels, load_labels
from .model import HEURISTIC_WEIGHTS, check_weights, save_result, score
from .overview import build_overview
from .precheck import blocking, precheck, summarize
from .profile import profile_dataset
from .question import QuestionSpec, apply_population, parse_filters
from .report import (build_summary, load_spec, read_result_extras, read_result_workbook, write_analysis_spec, write_result_workbook,
                     write_run_manifest, write_summary)
from .validation import validate

DATASET_ROOT = Path(os.environ.get("HR_DATASET_ROOT", "~/project_tpc/offpiste_synthetic_HR_data/output")).expanduser()
EXIT_BLOCKED = 2


def _add_predict_args(ap):
    ap.add_argument("--input", "--data", "--preset", dest="input", help="HR xlsx 파일, output/<preset> 폴더, 또는 프리셋 이름")
    ap.add_argument("--as-of", dest="as_of", help="기준일 YYYY-MM-DD (파일에 명시적 기준일이 없거나 충돌할 때 지정)")
    ap.add_argument("--labels", type=Path, help="라벨 CSV(employee_id,left). 없으면 워크북의 라벨 시트를 찾는다")
    ap.add_argument("--mode", choices=["auto", "model", "heuristic"], default=None, help="계산 방식(기본 auto)")
    ap.add_argument("--model-artifact", type=Path, help="train으로 만든 신뢰할 수 있는 고정 모델(model.joblib)")
    ap.add_argument("--weights", type=Path, help="규칙 가중치 덮어쓰기 JSON {신호: 가중치}")
    ap.add_argument("--spec", type=Path, help="이전 실행의 analysis_spec.json을 재사용(명시적 CLI 인자가 우선)")
    ap.add_argument("--top", type=int, default=None, help="요약에 넣을 상위 위험 직원 수(기본 10)")
    ap.add_argument("--target", help="관심 사건 (기본: 자발적 퇴사)")
    ap.add_argument("--horizon-days", type=int, help="요청 기간: 기준일 이후 N일 (기본 90)")
    ap.add_argument("--population", help="기준 모집단 설명")
    ap.add_argument("--filter", action="append", help="기준 모집단 제한 키=값 (본부, 팀, 직군, 직급, 고용형태, 재직상태, 직책). 반복 가능")
    ap.add_argument("--prompt", default="", help="사용자 요청 원문(기록용)")


def _out_dir(args, fallback_name: str) -> Path:
    out = args.out or (C.DEFAULT_OUT_ROOT / fallback_name)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_settings(args) -> dict:
    """명시적 CLI > 스펙 > 기본값."""
    spec = load_spec(args.spec) if getattr(args, "spec", None) else {}
    weights = None
    if getattr(args, "weights", None):
        weights = check_weights(json.loads(Path(args.weights).read_text(encoding="utf-8")))
    elif spec:
        full = spec.get("weights_full")
        if full:   # 저장된 전체 가중치를 그대로 복원(기본 가중치가 바뀌어도 같은 결과)
            weights = check_weights({k: v for k, v in full.items() if k in full and HEURISTIC_WEIGHTS.get(k) != v} or {}) or None
            weights = check_weights(full)
        elif spec.get("weights_override"):
            weights = check_weights(spec["weights_override"])
    mode = args.mode or spec.get("mode_requested") or "auto"
    top = args.top if args.top is not None else 10
    return dict(spec=spec, weights=weights, mode=mode, top=top)


def predict(args) -> tuple[Path | None, dict | None]:
    if not args.input:
        raise SystemExit("--input <HR.xlsx>가 필요합니다.")
    ds = load_any(args.input, DATASET_ROOT, as_of=getattr(args, "as_of", None))
    out = _out_dir(args, Path(args.input).stem)
    print(f"[input] {ds.kind}: {ds.source} · {ds.manifest['company']['name']} · 기준일 {ds.manifest.get('as_of') or '미확인'} · {ds.manifest['employee_count']}명")

    # 2a 입력 선검사(피처 계산 전)
    issues = precheck(ds)
    blocked = blocking(issues)
    (out / "diagnostics.json").write_text(json.dumps(dict(source=ds.source, as_of=ds.manifest.get("as_of"), issues=issues, blocked=bool(blocked)),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[2 선검사] {summarize(issues)}" + (" → 차단: " + "; ".join(i["message"] for i in blocked) if blocked else ""))
    if blocked:
        print(f"[중단] 해결 방법: " + " / ".join(i["resolution"] for i in blocked) + f" · 진단: {out / 'diagnostics.json'}")
        return None, dict(blocked=True, issues=issues, diagnostics=str(out / "diagnostics.json"))

    settings = _resolve_settings(args)
    # 1 질문 정의 (CLI > 스펙 > 기본값)
    q = QuestionSpec.from_args(args.target, args.horizon_days, args.population, parse_filters(args.filter), args.spec, args.prompt)
    q.as_of, q.as_of_source = ds.manifest["as_of"], ds.manifest.get("as_of_source")
    model_path = getattr(args, "model_artifact", None)
    saved_model = settings["spec"].get("model_reference")
    if not model_path and saved_model:
        import hashlib
        model_path = Path(saved_model["path"])
        if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest() != saved_model["sha256"]:
            raise ValueError("재사용 스펙의 모델 파일이 없거나 변경되었습니다. --model-artifact로 사용할 모델을 명시하세요.")
    if model_path and settings["mode"] == "heuristic":
        raise ValueError("--model-artifact와 --mode heuristic은 함께 사용할 수 없습니다.")
    if settings["mode"] == "model" and not model_path:
        raise ValueError("예측에는 --model-artifact가 필요합니다. 과거 스냅샷으로 train을 먼저 실행하세요.")
    labels = None if model_path else load_labels(ds, args.labels, q.target, q.horizon_days)
    q.analysis_mode = "diagnostic"   # predictive는 라벨이 있고 모델이 실제로 적용될 때만 아래에서 승격
    # 2b 데이터 이해
    f_all = build_features(ds)
    f, pop = apply_population(f_all, q)
    if pop["after"] == 0:
        raise SystemExit(f"기준 모집단 제한 {pop['filters']}에 해당하는 직원이 없습니다.")
    profile = profile_dataset(ds, f, set(f_all.employee_id.astype(str)))
    f.to_csv(out / "features.csv", index=False, encoding="utf-8-sig")
    miss = f.attrs.get("unavailable", [])
    exc = sum(f.attrs.get("excluded_after_as_of", {}).values())
    print(f"[2 데이터] 테이블 {sum(t['present'] for t in profile['tables'])}개 · 모집단 {pop['after']}/{pop['before']}명 · 신호 {len(f.columns)}열"
          + (f" · 계산 못 한 신호 {len(miss)}개" if miss else "") + f" · 기준일 이후 기록 {exc}행 제외" + (f" · 이슈 {len(profile['issues'])}건" if profile["issues"] else ""))
    if labels is not None:
        labels = labels[labels.index.isin(f.employee_id.astype(str))]
    d = describe_labels(labels, pop["after"])
    print("[labels] " + (f"있음 (행 {d['rows']}, 양성 {d['positives']}, 커버리지 {d['coverage_pct']}%)" if d["present"] else ("현재 라벨 미사용 → 고정 모델 적용" if model_path else "없음 → 진단 모드(규칙 기반 신호 지수)")))
    # 3 점수 + 가설
    if model_path:
        from .training import predict_frozen
        res = predict_frozen(f_all, model_path, q.as_of, q.target, q.horizon_days, ds.manifest["company"]["name"])
        # Population views retain company-wide ranks and bands.
        res.scores = res.scores[res.scores.employee_id.isin(f.employee_id)].copy()
        res.contrib = res.contrib.reindex(f.employee_id)
    else:
        if labels is not None and settings["mode"] != "heuristic":
            raise ValueError("라벨이 있는 파일로 현재 직원을 재학습하지 않습니다. train으로 과거 학습/미래 평가를 분리하세요. 진단만 하려면 --mode heuristic을 지정하세요.")
        res = score(f, None, "heuristic", settings["weights"])
    if model_path:
        import hashlib
        res.metrics["model_reference"] = dict(path=str(Path(model_path).resolve()), sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest())
    if res.mode == "model":
        q.analysis_mode = "predictive"
    print(f"[1 질문] {q.sentence()}")
    save_result(res, out)
    print(f"[score] 계산 {res.mode} · 밴드 {res.metrics.get('band_counts')}" + (f" · CV AUC {res.metrics['cv_auc']}" if "cv_auc" in res.metrics else ""))
    hyp = evaluate_hypotheses(f, res.scores, labels)
    print(f"[3 가설] {hyp['evidence_kind']}: " + ", ".join(f"{h['id']} {h['status']}" for h in hyp["hypotheses"]))
    # 4 검증
    val = validate(f, res, labels, q.as_of, q.as_of_source, settings["weights"], issues)
    print(f"[4 검증] 종합 {val['overall_ko']} · " + " · ".join(f"{c['id']} {c['status_ko']}" for c in val["checklist"]))
    # 5 행동 + 6 재사용
    summary = build_summary(f, res, ds.manifest, settings["top"], q.to_dict(), profile, hyp, val, pop)
    summary["precheck"] = issues
    write_summary(summary, out)
    xlsx = write_result_workbook(f, res, summary, ds.manifest, out)
    spec_path = write_analysis_spec(summary, res, settings["weights"], settings["mode"], out)
    inputs = [Path(ds.source)] + ([Path(args.labels)] if args.labels else [])
    write_run_manifest(out, inputs, spec_path, [i["message"] for i in issues if i["severity"] == "warn"],
                       dict(population=pop, analysis_mode=q.analysis_mode, calc_mode=res.mode, reused_spec=str(args.spec) if args.spec else None))
    print(f"[5 행동] 회사 신호 {len(summary['company_signals'])}개 · 조치 {len(summary['actions'])}개 · 팀별 확인사항 {len(summary['unit_actions'])}팀")
    print(f"[6 재사용] {spec_path} · run_manifest.json")
    print(f"[result] {xlsx}")
    print(f"완료: {summary['company']} · 기준일 {summary['as_of']} · {summary['n_employees']}명 · "
          f"높음 {summary['bands']['높음']} / 주의 {summary['bands']['주의']} / 낮음 {summary['bands']['낮음']} · {summary['mode_label']}")
    return xlsx, summary


def dashboard(args) -> Path | None:
    result = args.result
    if result is None:
        if not args.input:
            raise SystemExit("--result <퇴사위험_결과.xlsx> 또는 --data <HR.xlsx>가 필요합니다.")
        print("[dashboard] 결과 파일이 없어 진단을 먼저 수행합니다.")
        result, summary = predict(args)
        if result is None:
            return None
    features, scores, meta = read_result_workbook(result)
    out = _out_dir(args, Path(result).parent.name)
    raw = None
    if args.input:
        try:
            raw = load_any(args.input, DATASET_ROOT, as_of=getattr(args, "as_of", None))
            print(f"[data] 원본 데이터 로드: {raw.source}")
        except Exception as e:  # noqa: BLE001 - 원본은 선택 입력
            print(f"[data] 원본 데이터를 읽지 못해 결과 파일만으로 현황을 만듭니다: {e}")
    manifest = meta["manifest"]
    if raw and (str(raw.manifest.get("as_of")) != str(manifest.get("as_of")) or raw.manifest["company"]["name"] != manifest["company"]["name"]):
        raise ValueError("대시보드 원본과 결과의 회사·기준일이 다릅니다.")
    ov = build_overview(features, str(manifest.get("as_of")), raw)
    extras = read_result_extras(result)
    p = build_dashboard(features, scores, meta["metrics"], manifest, ov, out / "dashboard.html", meta["unavailable"], extras)
    print(f"[dashboard] {p} · 직원 목록 중심 화면 (원본 {'포함: 필터 연동 근무·설문' if raw else '없음'})")
    if getattr(args, "share", False):
        ps = build_dashboard(features, scores, meta["metrics"], manifest, ov, out / "dashboard_share.html", meta["unavailable"], extras, share=True)
        print(f"[dashboard] {ps} · 공유용(이름·사번 제거, 가명 P-001…)")
    if args.open:
        if sys.platform == "win32":
            os.startfile(p)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open" if sys.platform.startswith("linux") else "open", str(p)])
    return p


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train", help="과거 스냅샷 학습·후보 비교·미래 기간 평가 → 고정 모델")
    pt.add_argument("--manifest", type=Path, required=True)
    pt.add_argument("--out", type=Path, required=True)
    p1 = sub.add_parser("predict", help="진단/예측 → 결과 엑셀")
    _add_predict_args(p1)
    p1.add_argument("--out", type=Path)
    p2 = sub.add_parser("dashboard", help="결과 엑셀(+원본) → 대시보드 HTML")
    _add_predict_args(p2)
    p2.add_argument("--result", type=Path, help="/predict_attrition이 만든 퇴사위험_결과.xlsx")
    p2.add_argument("--out", type=Path)
    p2.add_argument("--open", action="store_true")
    p2.add_argument("--share", action="store_true", help="공유용 dashboard_share.html도 생성(이름·사번 제거)")
    p3 = sub.add_parser("run", help="predict + dashboard")
    _add_predict_args(p3)
    p3.add_argument("--out", type=Path)
    p3.add_argument("--open", action="store_true")
    p3.add_argument("--share", action="store_true", help="공유용 dashboard_share.html도 생성(이름·사번 제거)")
    args = ap.parse_args(argv)

    if args.cmd == "train":
        from .training import train
        path, report = train(args.manifest, args.out)
        print(f"모델 {path} · 선택 {report['selected']} · 미래 기간 평균 Precision@K {report['future_evaluation']['mean_precision_at_k']:.3f}")
        return
    if args.cmd == "predict":
        xlsx, info = predict(args)
        if xlsx is None:
            sys.exit(EXIT_BLOCKED)
    elif args.cmd == "dashboard":
        if dashboard(args) is None:
            sys.exit(EXIT_BLOCKED)
    else:
        xlsx, _ = predict(args)
        if xlsx is None:
            sys.exit(EXIT_BLOCKED)
        args.result = xlsx
        dashboard(args)


def main(argv=None):
    try:
        return _main(argv)
    except (ValueError, FileNotFoundError, ImportError) as error:
        print(json.dumps({"error": str(error), "status": "blocked"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(EXIT_BLOCKED) from None


if __name__ == "__main__":
    main()
