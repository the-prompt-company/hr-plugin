#!/usr/bin/env python
"""HR xlsx 한 파일 → 퇴사 위험 예측 → 퇴사위험_결과.xlsx (+ summary.json).

    python predict_attrition.py --input <HR.xlsx> --out <출력 폴더> [--mode auto|model|heuristic] [--labels l.csv] [--weights w.json] [--top 10]

동봉된 hrattrition 패키지(zip 안 scripts/hrattrition, 저장소에서는 engine/hrattrition)를 쓴다.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parents[2] / "engine"):
    if (cand / "hrattrition").is_dir():
        sys.path.insert(0, str(cand))
        break

from hrattrition.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--train-manifest" in args:
        args[args.index("--train-manifest")] = "--manifest"
        main(["train", *args])
    else:
        main(["predict", *args])
