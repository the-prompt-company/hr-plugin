"""Construct outcomes from event dates, type and complete observation windows.

A binary left column is not a sufficient label contract. Non-target exits are
excluded from voluntary-exit training (competing event), never silent negatives.
"""
from pathlib import Path
import pandas as pd

TARGETS = {"자발적 퇴사", "전체 퇴사"}
VOLUNTARY = {"자발적", "자발적 퇴사", "voluntary"}
OTHER = {"비자발적", "비자발적 퇴사", "involuntary", "계약만료", "계약 만료", "정년", "retirement"}


def build_labels(frame, as_of, target="자발적 퇴사", horizon_days=90):
    if target not in TARGETS or horizon_days <= 0:
        raise ValueError("target은 자발적 퇴사/전체 퇴사, horizon_days는 양수여야 합니다.")
    df = frame.rename(columns={"사번": "employee_id", "퇴사일": "left_date", "퇴사유형": "left_type", "관측종료일": "observed_until"}).copy()
    required = {"employee_id", "left_date", "left_type", "observed_until"}
    if not required <= set(df):
        raise ValueError("기간별 라벨에는 employee_id, left_date, left_type, observed_until이 필요합니다. left(0/1)만으로 학습하지 않습니다.")
    ids = df.employee_id.astype('string').str.strip()
    if ids.isna().any() or ids.str.lower().isin(['', 'none', 'nan', 'nat']).any() or ids.duplicated().any():
        raise ValueError("라벨 사번은 비어 있지 않은 고유 문자열이어야 합니다. 재입사는 고용 에피소드별로 분리하세요.")
    dates = {}
    for col in ('left_date', 'observed_until'):
        raw = df[col].replace('', None)
        dates[col] = pd.to_datetime(raw, errors='coerce')
        if (raw.notna() & dates[col].isna()).any() or df.attrs.get('date_parse_errors', {}).get(col):
            raise ValueError(f"라벨 날짜를 해석할 수 없습니다: {col}")
    left, observed = dates['left_date'], dates['observed_until']
    if observed.isna().any() or (left.notna() & (left > observed)).any():
        raise ValueError("관측 종료일이 없거나 퇴사일보다 이릅니다.")
    types = df.left_type.fillna('').astype(str).str.strip().str.lower()
    if (left.isna() & types.ne('')).any():
        raise ValueError("퇴사 유형은 있지만 퇴사일이 없습니다.")
    if 'left' in df and (left.isna() & df.left.astype(str).str.lower().isin(['1', '1.0', 'true', '예', '퇴사'])).any():
        raise ValueError("퇴사 라벨 1인 행에 퇴사일이 없습니다.")
    start, end = pd.Timestamp(as_of), pd.Timestamp(as_of) + pd.Timedelta(days=horizon_days)
    y = pd.Series(float('nan'), index=df.index, dtype=float)
    status = pd.Series('관측 미완료', index=df.index)
    at_risk = (left.isna() | left.gt(start)) & observed.ge(start)
    complete = at_risk & observed.ge(end)
    in_window = at_risk & left.gt(start) & left.le(end)
    y.loc[complete & ~in_window] = 0
    status.loc[complete & ~in_window] = '기간 내 대상 사건 없음'
    if target == '전체 퇴사':
        event = in_window
    else:
        known = types.isin(VOLUNTARY | OTHER)
        status.loc[in_window & ~known] = '퇴사 유형 불명'
        status.loc[in_window & types.isin(OTHER)] = '경쟁 사건 제외'
        event = in_window & types.isin(VOLUNTARY)
    y.loc[event] = 1
    status.loc[event] = '대상 퇴사'
    status.loc[~at_risk] = '기준일 모집단 아님'
    y.index = ids.astype(str)
    y.name = 'left'
    y.attrs = dict(target=target, horizon_days=horizon_days, as_of=start.date().isoformat(),
                   statuses=dict(zip(ids.astype(str), status)), excluded=int(y.isna().sum()),
                   label_window_end=end.date().isoformat())
    return y


def load_labels(ds, path: Path | None = None, target="자발적 퇴사", horizon_days=90):
    if path is not None:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    elif ds.has('labels'):
        df = ds['labels']
    else:
        return None
    return build_labels(df, ds.manifest['as_of'], target, horizon_days)


def describe_labels(s, n_employees):
    if s is None:
        return dict(present=False, rows=0, positives=0, coverage_pct=0.0)
    return dict(present=True, rows=int(s.notna().sum()), positives=int(s.sum()),
                excluded=int(s.isna().sum()), coverage_pct=round(100*s.notna().sum()/max(n_employees, 1), 1))
