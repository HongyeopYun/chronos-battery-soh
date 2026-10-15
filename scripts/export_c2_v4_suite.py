"""Finalize a verified remote run and export results and selected weights separately."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--skip-weights-zip',action='store_true');a=p.parse_args()
    out=a.out.resolve();verification=json.loads((out/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='passed' and verification['checkpoints_sha256_checked_locally']==117
    assert (out/'report.md').exists()
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(out/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        sources=[*(root/'src').glob('*.py'),root/'scripts/run_c2_v4_suite.py',
                 root/'scripts/verify_report_c2_v4_suite.py',Path(__file__),root/'tests/test_c2_v4_experiments.py']
        for path in sorted(sources):z.write(path,path.relative_to(root))
    artifacts={str(path.relative_to(out)).replace('\\','/'):dict(sha256=digest(path),bytes=path.stat().st_size)
        for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ('complete.json','remote_artifact_manifest.json','status.json')}
    write(out/'remote_artifact_manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),root=str(out),artifacts=artifacts))
    complete=dict(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(out/'protocol.json'),verification_sha256=digest(out/'verification.json'),
        remote_artifact_manifest_sha256=digest(out/'remote_artifact_manifest.json'),
        selected_models=117,report='report.md',exporter_sha256=digest(Path(__file__)),
        report_builder_sha256=digest(Path(__file__).with_name('verify_report_c2_v4_suite.py')))
    write(out/'complete.json',complete);write(out/'status.json',complete)
    results_zip=out.parent/(out.name+'-results.zip')
    with zipfile.ZipFile(results_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix!='.pt':z.write(path,path.relative_to(out))
    print(json.dumps(dict(results_zip=str(results_zip),bytes=results_zip.stat().st_size)),flush=True)
    if not a.skip_weights_zip:
        weights_zip=out.parent/(out.name+'-weights.zip')
        with zipfile.ZipFile(weights_zip,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for path in sorted(out.rglob('weights.pt')):z.write(path,path.relative_to(out))
        print(json.dumps(dict(weights_zip=str(weights_zip),bytes=weights_zip.stat().st_size)),flush=True)


if __name__=='__main__':main()
