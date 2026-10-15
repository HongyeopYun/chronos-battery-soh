"""Transfer finished raw GPU artifacts for independent local audit/reporting."""
import argparse,hashlib,json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    finish=json.loads((out/'execution_finished.json').read_text())
    assert finish['predictions_sha256']==digest(out/'predictions.csv')
    assert finish['protocol_sha256']==digest(out/'protocol.json')
    protocol=json.loads((out/'protocol.json').read_text())
    assert len(list((out/'checkpoints').glob('*/weights.pt')))==len(protocol['cases'])
    lock=json.loads((out/'selection_lock.json').read_text())
    for path,sha in lock['selected_sha256'].items():
        assert digest(out/path)==sha
        selected=json.loads((out/path).read_text())
        assert selected['weights_sha256']==digest((out/path).parent/'weights.pt')
    archive=out.with_suffix('.raw.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=3) as z:
        for item in out.rglob('*'):
            if item.is_file():z.write(item,item.relative_to(out).as_posix())
    print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive))))
