"""Regression tests for target semantics, leakage, time splits and frozen inference."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hrattrition.data import coerce, load_any, load_roster_frame
from hrattrition.features import build_features
from hrattrition.labels import build_labels
from hrattrition.precheck import precheck
from hrattrition.training import train, predict_frozen, metrics_at_k


def events():
    return pd.DataFrame(dict(employee_id=['a','b','c','d','e'],left_date=['2026-02-10','2026-05-10',None,'2026-02-10',None],left_type=['자발적','자발적','','계약만료',''],observed_until=['2026-08-01']*4+['2026-02-01']))


def test_labels_use_horizon_type_and_maturity():
    df=events();a=build_labels(df,'2026-01-01','자발적 퇴사',90);b=build_labels(df,'2026-01-01','전체 퇴사',180)
    assert a['a']==1 and a['b']==0 and pd.isna(a['d']) and pd.isna(a['e'])
    assert b['a']==b['b']==b['d']==1 and b['c']==0 and pd.isna(b['e'])
    df['left']=0 # provided stale bits must not override dates
    assert build_labels(df,'2026-01-01')['a']==1
    with pytest.raises(ValueError):build_labels(df[['employee_id','left']],'2026-01-01')
    with pytest.raises(ValueError):build_labels(df,'2026-01-01',horizon_days=0)


def test_invalid_event_dates_not_hidden_by_coercion():
    df=events();df.loc[0,'left_date']='bad';df=coerce(df)
    with pytest.raises(ValueError):build_labels(df,'2026-01-01')


def test_roster_future_dates_actually_removed():
    df=pd.DataFrame({'사번':['A','B'],'입사일':['2020-01-01']*2,'마지막인상일':['2027-01-01','2026-01-01'],'최근승진일':['2027-01-01','2026-01-01']})
    ds=load_roster_frame(df,'sample');ds.manifest.update(as_of='2026-04-30',as_of_source='user')
    f=build_features(ds).set_index('employee_id')
    assert pd.isna(f.at['A','months_since_raise']) and pd.isna(f.at['A','months_since_promotion'])
    assert f.at['B','months_since_raise']>0


def test_parse_errors_survive_file_loader(tmp_path):
    df=pd.DataFrame({'사번':['A'],'입사일':['2020-01-01'],'마지막인상일':['not-date']})
    p=tmp_path/'r.xlsx';df.to_excel(p,index=False);ds=load_any(str(p),as_of='2026-04-30')
    assert any(x['code']=='DATE_PARSE_FAIL' for x in precheck(ds))
    assert build_features(ds).months_since_raise.isna().all()


def test_recorded_after_snapshot_is_excluded():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from make_fixture import make
    from hrattrition.data import Dataset
    tables={k.removeprefix('hr_').removeprefix('v_'):coerce(v) for k,v in make(60,7).items() if k.startswith(('hr_','v_'))}
    ds=Dataset('memory',dict(as_of='2026-04-30',company={'name':'test'}),tables)
    baseline=build_features(ds)
    df=ds.tables['compensation_history'].copy()
    eid=df.employee_id.iloc[0];mask=df.employee_id.eq(eid);df.loc[mask,'recorded_at']=pd.Timestamp('2027-01-01');ds.tables['compensation_history']=df
    f=build_features(ds).set_index('employee_id')
    assert pd.isna(f.at[eid,'band_position'])
    assert f.attrs['excluded_after_as_of']['compensation_history.recorded']>=1


def test_metrics_k_clips_and_no_positives():
    m=metrics_at_k([1,0,1],[.9,.8,.1],20)
    assert m['k']==3 and m['hits']==2 and m['recall_at_k']==1
    assert metrics_at_k([0,0],[.2,.1],1)['recall_at_k'] is None


@pytest.fixture(scope='module')
def temporal(tmp_path_factory):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    from make_temporal_fixture import generate
    root=tmp_path_factory.mktemp('temporal');manifest=generate(root,120)
    path,report=train(manifest,root/'model')
    return root,manifest,path,report


def test_time_split_candidate_selection_and_export(temporal):
    root,manifest,path,r=temporal
    assert len(r['candidates'])==5
    assert r['split']['train_end'] < r['split']['validation_start'] < r['split']['test_start']
    assert r['future_evaluation']['periods'] and r['source_hashes'] and r['synthetic']
    assert (path.parent/'candidate_comparison.csv').exists()
    assert (path.parent/'future_predictions.csv').exists()
    assert all('test' not in c for c in r['candidates'])


def test_frozen_inference_no_training_and_reject_mismatch(temporal):
    root,manifest,path,r=temporal
    ds=load_any(str(root/'current.xlsx'));f=build_features(ds)
    a=predict_frozen(f,path,'2026-04-30','자발적 퇴사',90,ds.manifest['company']['name'])
    b=predict_frozen(f.iloc[::-1],path,'2026-04-30','자발적 퇴사',90,ds.manifest['company']['name'])
    pd.testing.assert_series_equal(a.scores.set_index('employee_id').risk_score.sort_index(),b.scores.set_index('employee_id').risk_score.sort_index())
    assert a.model is None and len(a.scores)==120 and a.metrics['evaluation']['model_id']==r['model_id']
    for target,days,date in [('전체 퇴사',90,'2026-04-30'),('자발적 퇴사',180,'2026-04-30'),('자발적 퇴사',90,'2025-04-30')]:
        with pytest.raises(ValueError):predict_frozen(f,path,date,target,days,ds.manifest['company']['name'])


def test_test_labels_do_not_choose_candidate(temporal,tmp_path):
    # A manifest reusing training and validation but reversing only future labels.
    root,manifest,path,r=temporal
    spec=json.loads(manifest.read_text())
    for item in spec['snapshots']:
        source=root/item['input'];item['input']=str(source)
        if '2025-' in source.name:
            sheets=pd.read_excel(source,sheet_name=None)
            lab=sheets['퇴사라벨'];asof=pd.Timestamp(source.stem.removeprefix('snapshot-'))
            old=build_labels(lab,asof)
            # deterministic alternating future events, independent of previous labels
            for i in lab.index:
                lab.at[i,'left_date']=asof+pd.Timedelta(days=30) if i%2 else pd.NaT
                lab.at[i,'left_type']='자발적' if i%2 else ''
                lab.at[i,'left']=int(i%2)
            dest=tmp_path/source.name
            with pd.ExcelWriter(dest,engine='openpyxl') as w:
                for name,df in sheets.items():df.to_excel(w,sheet_name=name,index=False)
            item['input']=str(dest)
    changed=tmp_path/'manifest.json';changed.write_text(json.dumps(spec))
    _,r2=train(changed,tmp_path/'model')
    assert r2['selected']==r['selected']
    assert r2['candidates']==r['candidates']


def test_no_snapshot_date_override(temporal):
    root,*_=temporal
    with pytest.raises(ValueError):load_any(str(root/'current.xlsx'),as_of='2025-04-30')
