"""Verify and install locally downloaded release archives inside this package."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archives',nargs='+',type=Path);args=ap.parse_args()
    m=json.loads((HERE/'weights_manifest.json').read_text())
    for path in args.archives:
        asset=next((x for x in m['assets'] if x['name']==path.name),None)
        assert asset is not None,f'Unknown archive {path.name}'
        assert path.stat().st_size==asset['bytes'] and digest(path)==asset['sha256'],'Archive checksum mismatch'
        wanted={r['file']:r for r in m['models'] if r['asset']==path.name}
        with zipfile.ZipFile(path) as z:
            assert set(z.namelist())==set(wanted),'Unexpected archive contents'
            for name,rec in wanted.items():
                target=(HERE/name).resolve()
                assert target.is_relative_to(HERE) and target!=HERE,'Unsafe archive path'
                if target.exists():
                    assert digest(target)==rec['sha256'],'Existing model differs; no overwrite performed'
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_name(target.name+'.installing')
                assert not temp.exists(),'An interrupted installation file already exists'
                h=hashlib.sha256()
                with z.open(name) as src,temp.open('wb') as dst:
                    for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b);dst.write(b)
                assert h.hexdigest()==rec['sha256'] and temp.stat().st_size==rec['bytes']
                temp.replace(target)
        print('Installed',path.name,len(wanted),'models',flush=True)


if __name__=='__main__':main()
