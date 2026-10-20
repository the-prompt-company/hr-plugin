"""Generate explicitly synthetic, time-separated pipeline fixtures (not performance evidence)."""
import argparse
import json
from pathlib import Path
import pandas as pd
from make_fixture import make, AS_OF


def generate(out, n=186):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dates=['2023-01-31','2023-04-30','2023-07-31','2024-01-31','2024-04-30','2025-01-31','2025-04-30','2026-04-30']
    snapshots=[]
    for i,date in enumerate(dates):
        sheets=make(n,7+i)
        delta=pd.Timestamp(date)-pd.Timestamp(AS_OF)
        month_delta=(pd.Timestamp(date).year-AS_OF.year)*12+pd.Timestamp(date).month-AS_OF.month
        for name,df in sheets.items():
            for col in df:
                if col=='pay_month':
                    df[col]=df[col].map(lambda x:str(pd.Period(x,freq='M')+month_delta))
                elif col.endswith(('_date','_at','_from','_to')) or col in ('as_of','week_start','closed_at','observed_until'):
                    df[col]=pd.to_datetime(df[col],errors='coerce')+delta
            if name=='개요':
                df.loc[df.iloc[:,0]=='기준일',df.columns[1]]=date
                df.loc[df.iloc[:,0]=='관측 종료일',df.columns[1]]=(pd.Timestamp(date)+pd.Timedelta(days=180)).date().isoformat()
        if i==len(dates)-1:
            sheets.pop('퇴사라벨')
            filename='current.xlsx'
        else:
            filename=f'snapshot-{date}.xlsx'
            snapshots.append(dict(input=filename))
        with pd.ExcelWriter(out/filename,engine='openpyxl') as writer:
            for name,df in sheets.items():df.to_excel(writer,sheet_name=name,index=False)
    manifest=dict(target='자발적 퇴사',horizon_days=90,top_k=20,validation_start='2024-01-01',test_start='2025-01-01',observed_through='2025-08-01',synthetic=True,snapshots=snapshots)
    (out/'history.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return out/'history.json'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--n',type=int,default=186);a=ap.parse_args();print(generate(a.out,a.n))
