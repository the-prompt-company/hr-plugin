"""1단계 질문 정의: 요청한 질문과 지금 데이터로 답할 수 있는 질문을 구분해 기록한다(P0-6).

analysis_mode(제품 수준): diagnostic(현재 신호 진단) | change(시점 비교) | predictive(검증된 기간별 예측).
라벨이 없으면 predictive가 될 수 없고, 진단 모드에서는 "N일 내 퇴사 예측"이라고 출력하지 않는다.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import config as C

DEFAULT_TARGET = "자발적 퇴사"
DEFAULT_HORIZON_DAYS = 90
DEFAULT_POPULATION = "기준일 현재 재직자 전원(휴직자 포함)"
FILTERABLE = ("division_name", "org_unit_name", "job_family", "grade_title", "employment_type", "employment_status", "role_title")
FILTER_ALIASES = {"본부": "division_name", "팀": "org_unit_name", "부서": "org_unit_name", "직군": "job_family", "직급": "grade_title",
                  "고용형태": "employment_type", "재직상태": "employment_status", "직책": "role_title"}


@dataclass
class QuestionSpec:
    target: str = DEFAULT_TARGET                 # 관심 사건: 자발적 퇴사 / 전체 퇴사 …
    horizon_days: int = DEFAULT_HORIZON_DAYS     # 요청한 예측 기간(기준일 이후 N일)
    population: str = DEFAULT_POPULATION         # 기준 모집단 설명
    filters: dict = field(default_factory=dict)  # 기준 모집단 제한(표시 필터와 구분)
    as_of: str | None = None
    as_of_source: str | None = None
    raw_prompt: str = ""
    analysis_mode: str = "diagnostic"            # diagnostic | change | predictive (라벨·검증 조건이 갖춰질 때만 predictive)

    def population_text(self) -> str:
        return self.population + (" 중 " + ", ".join(f"{k}={v}" for k, v in self.filters.items()) if self.filters else "")

    def requested_question(self) -> str:
        return f"기준일 {self.as_of or '(파일 기준)'} 이후 {self.horizon_days}일 이내 {self.target} 위험을 {self.population_text()}에 대해 알고 싶다."

    def answerable_question(self) -> str:
        if self.analysis_mode == "predictive":
            return (f"기준일 {self.as_of} 이후 {self.horizon_days}일 이내 {self.target} 위험 순위를 {self.population_text()}에 대해 예측한다"
                    f"(과거 데이터로 학습·시간 분리 평가한 고정 모델. 현재 직원의 결과 라벨은 학습에 사용하지 않음).")
        if self.analysis_mode == "change":
            return f"두 기준일 사이 {self.population_text()}의 퇴사 관련 신호 변화를 비교한다."
        return (f"기준일 {self.as_of or '(파일 기준)'} 현재 {self.population_text()}의 퇴사 관련 신호를 진단한다. "
                f"검증된 예측 모델을 적용하지 않아 '{self.horizon_days}일 이내 {self.target}' 여부는 예측하지 않는다.")

    def sentence(self) -> str:
        return self.answerable_question()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["requested_question"] = self.requested_question()
        d["answerable_question"] = self.answerable_question()
        d["sentence"] = self.sentence()
        return d

    @classmethod
    def from_args(cls, target=None, horizon_days=None, population=None, filters=None, spec_path: Path | None = None, raw_prompt="",
                  analysis_mode=None) -> "QuestionSpec":
        """우선순위: 명시적 인자 > 재사용 스펙 > 기본값. 인자가 None이면 '지정 안 함'으로 본다."""
        base = {}
        if spec_path:
            base = json.loads(Path(spec_path).read_text(encoding="utf-8")).get("question", {}) or {}
            base = {k: v for k, v in base.items() if k in cls.__dataclass_fields__ and k not in ("as_of", "as_of_source", "raw_prompt")}
        q = cls(**base)
        if target:
            q.target = target
        if horizon_days is not None:
            if int(horizon_days) <= 0:
                raise ValueError("예측 기간은 양의 정수여야 합니다.")
            q.horizon_days = int(horizon_days)
        if population:
            q.population = population
        if filters:
            q.filters = dict(filters)
        if raw_prompt:
            q.raw_prompt = raw_prompt
        if analysis_mode:
            if analysis_mode not in C.ANALYSIS_MODES:
                raise ValueError(f"analysis_mode는 {C.ANALYSIS_MODES} 중 하나")
            q.analysis_mode = analysis_mode
        return q


def parse_filters(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--filter 형식은 키=값 입니다: {it}")
        k, v = it.split("=", 1)
        k = FILTER_ALIASES.get(k.strip(), k.strip())
        if k not in FILTERABLE:
            raise ValueError(f"필터 가능한 키: {', '.join(FILTERABLE)} (별칭: {', '.join(FILTER_ALIASES)}). 민감 속성으로는 거를 수 없습니다.")
        out[k] = v.strip()
    return out


def apply_population(features: pd.DataFrame, q: QuestionSpec) -> tuple[pd.DataFrame, dict]:
    before = len(features)
    f = features
    for k, v in q.filters.items():
        if k in f:
            f = f[f[k].astype(str) == str(v)]
    f = f.copy()
    f.attrs = dict(features.attrs)
    return f, dict(before=before, after=int(len(f)), filters=dict(q.filters),
                   note="기준 모집단 제한. 점수의 표준화도 이 모집단 안에서 이뤄진다. 결과를 좁혀 보기만 하려면 표시 필터(대시보드·후속 질의)를 쓴다.")
