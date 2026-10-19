"""claude.ai 업로드용 스킬 zip 두 개 생성. 각 zip에 공용 엔진(engine/hrattrition)을 scripts/hrattrition으로 동봉한다.

    python make_skill_zip.py            → dist/predict_attrition.zip, dist/make_dashboard.zip
    python make_skill_zip.py --check    → zip을 임시 폴더에 풀어 저장소 engine 없이 단독 실행되는지, 엔진 버전이 일치하는지 검사

P0-5: 버전 단일 원천은 engine/hrattrition/__init__.py의 __version__. plugin.json 버전이 다르면 빌드를 거부한다.
"""
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine" / "hrattrition"
SKILLS = ("predict_attrition", "make_dashboard")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
DATA_EXT = {".xlsx", ".xlsm", ".xls", ".csv", ".parquet"}   # 실제 HR 데이터가 섞여 들어가면 안 된다


def engine_version() -> str:
    return re.search(r'__version__\s*=\s*"([^"]+)"', (ENGINE / "__init__.py").read_text(encoding="utf-8")).group(1)


def _ok(p: Path) -> bool:
    return p.is_file() and not any(part in SKIP_DIRS or part.endswith(".egg-info") for part in p.parts) and p.suffix != ".pyc"


def check_versions() -> str:
    v = engine_version()
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    if plugin.get("version") != v:
        raise SystemExit(f"버전 불일치: engine {v} vs plugin.json {plugin.get('version')} — plugin.json을 맞춘 뒤 다시 빌드한다.")
    return v


def build() -> list[Path]:
    v = check_versions()
    (ROOT / "dist").mkdir(exist_ok=True)
    outs = []
    for name in SKILLS:
        src = ROOT / "skill" / name
        out = ROOT / "dist" / f"{name}.zip"
        n = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(src.rglob("*")):
                if _ok(p):
                    if p.suffix.lower() in DATA_EXT:
                        raise SystemExit(f"데이터 파일이 스킬 폴더에 있습니다: {p} — 배포 패키지에 포함할 수 없습니다.")
                    z.write(p, Path(name) / p.relative_to(src))
                    n += 1
            for p in sorted(ENGINE.rglob("*")):
                if _ok(p):
                    z.write(p, Path(name) / "scripts" / "hrattrition" / p.relative_to(ENGINE))
                    n += 1
            z.writestr(str(Path(name) / "VERSION"), v + "\n")
        print(f"{out} ← {n + 1} files, {out.stat().st_size / 1024:.0f} KB, engine {v}")
        outs.append(out)
    return outs


def check_zips(sample_xlsx: Path | None) -> None:
    """두 zip을 임시 폴더에 풀어 저장소 engine 없이 실행한다(단독 실행·버전 일치 검사)."""
    v = engine_version()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for name in SKILLS:
            with zipfile.ZipFile(ROOT / "dist" / f"{name}.zip") as z:
                z.extractall(tdp)
            ver = (tdp / name / "VERSION").read_text().strip()
            init = (tdp / name / "scripts" / "hrattrition" / "__init__.py").read_text(encoding="utf-8")
            assert ver == v and f'"{v}"' in init, f"{name}: 버전 불일치 {ver} / {v}"
            print(f"[check] {name}: engine {ver} OK")
        if sample_xlsx and sample_xlsx.exists():
            out = tdp / "out"
            r = subprocess.run([sys.executable, str(tdp / "predict_attrition" / "scripts" / "predict_attrition.py"), "--input", str(sample_xlsx), "--mode", "heuristic", "--out", str(out)],
                               capture_output=True, text=True, env={**__import__("os").environ, "PYTHONUTF8": "1"})
            assert r.returncode == 0, r.stderr
            r = subprocess.run([sys.executable, str(tdp / "make_dashboard" / "scripts" / "make_dashboard.py"), "--result", str(out / "퇴사위험_결과.xlsx"), "--out", str(out)],
                               capture_output=True, text=True, env={**__import__("os").environ, "PYTHONUTF8": "1"})
            assert r.returncode == 0, r.stderr
            print(f"[check] zip 단독 실행 OK ({sample_xlsx.name})")


if __name__ == "__main__":
    build()
    if "--check" in sys.argv:
        check_zips(ROOT / "fixtures" / "hr_fixture_186.xlsx")
