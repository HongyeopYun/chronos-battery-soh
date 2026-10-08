"""Verify the completed migration and refresh compact reports/indexes."""
import json
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows
from src.battery_model_zeroshot import verify_result, write_report, write_comparison, write_model_suite
from src.battery_zeroshot_minimal import verify_compact_result, write_compact_report, write_suite_index, GROUPS
from scripts.run_calce_zeroshot import write_comparison as calce_comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true',help='Refresh only already migrated packages')
    args=parser.parse_args()
    count=0
    for tag in ['c2-battery','t5-battery']:
        for path in sorted((ROOT/'results'/tag).rglob(f'{tag}_run.json')):
            if args.partial and not json.loads(path.read_text(encoding='utf-8')).get('r2_updated_at'):
                continue
            m=verify_result(path.parent,tag)
            cfg=ZeroShotConfig(**m['config'])
            coverage_path=path.parent/f'{tag}_coverage.csv'
            coverage=pd.read_csv(coverage_path,keep_default_na=False) if coverage_path.exists() else None
            summary=pd.read_csv(path.parent/f'{tag}_summary.csv')
            write_report(path.parent,tag,cfg,summary,m['figures'],m.get('run_scope','pilot'),coverage)
            count+=1
    compact_groups=[]
    for group in GROUPS:
        directory=ROOT/'results/t5-battery'/group
        if args.partial and not json.loads((directory/'run.json').read_text(encoding='utf-8')).get('r2_updated_at'):
            continue
        m=verify_compact_result(directory)
        cfg=ZeroShotConfig(**m['config'])
        _,coverage,_,_=prepare_battery_windows(m['bundle'],cfg)
        write_compact_report(directory,cfg,coverage,pd.read_csv(directory/'summary.csv'),m['figures'])
        count+=1
        compact_groups.append(group)
    write_model_suite(ROOT/'results/c2-battery/full','c2-battery',GROUPS)
    write_suite_index(ROOT/'results/t5-battery',compact_groups)
    write_comparison()
    calce_comparison(ROOT/'results/t5-c2-battery-comparison/calce64-16')
    # A single small JSON is convenient for presentation synchronization.
    tables={}
    for tag in ['c2-battery','t5-battery']:
        for path in sorted((ROOT/'results'/tag).rglob('*summary.csv')):
            frame=pd.read_csv(path)
            if 'group' in frame:
                continue
            if path.name=='summary.csv' and (path.parent/'run.json').exists() and not json.loads((path.parent/'run.json').read_text(encoding='utf-8')).get('r2_updated_at'):
                continue
            if 'r2' in frame:
                if 'phase' in frame:
                    frame=frame[(frame.phase=='all') | frame.model.isin(['t5-battery','c2-battery'])]
                fields=[k for k in ['model','phase','rmse','r2','wql','picp80','r2_scope'] if k in frame]
                tables[str(path.relative_to(ROOT)).replace('\\','/')]=json.loads(frame[fields].to_json(orient='records'))
    (ROOT/'results/window-r2/presentation_scores.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'WINDOW_R2_VERIFIED {count} runs')


if __name__=='__main__':
    main()
