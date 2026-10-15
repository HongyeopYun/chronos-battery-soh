"""Bundle verified results, all heads, and exact source snapshots for transfer."""
import argparse
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.battery_zeroshot import sha256
from scripts.run_c2_v4_suite import read,write,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();archive=a.archive.resolve()
    assert read(out/'complete.json')['status']=='complete'
    assert read(out/'verification_gpu.json')['status']=='passed'
    protocol=read(out/'protocol.json')
    assert all(sha256(ROOT/k)==v for k,v in protocol['source_sha256'].items())
    source_paths=set(protocol['source_sha256'])|{'scripts/report_c2_rul_learning_curve.py','scripts/export_c2_rul_learning_curve.py',
        'tests/test_c2_rul_learning_curve.py','tests/test_c2_v4_experiments.py'}
    with ZipFile(out/'source_snapshot.zip','w',ZIP_DEFLATED) as z:
        for rel in sorted(source_paths):z.write(ROOT/rel,rel)
    files={p.relative_to(out).as_posix():dict(sha256=sha256(p),size=p.stat().st_size)
           for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
    write(out/'artifact_manifest.json',dict(created_at=now(),files=files,weights=sum(k.endswith('/weights.pt') for k in files)))
    archive.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=3) as z:
        for rel in sorted([*files,'artifact_manifest.json']):z.write(out/rel,rel)
    print(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive),files=len(files)+1),flush=True)


if __name__=='__main__':main()
