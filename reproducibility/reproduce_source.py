"""Rerun a bounded source comparison in a separate output directory."""
import argparse
from execution_lock import hold_lock
import json
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
CODE=HERE/'analysis/experiments'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark',choices=['deeplab','inputs'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--data-root',type=Path)
    ap.add_argument('--evaluate',action='store_true',help='Evaluate completed fits without training')
    args=ap.parse_args()
    out=args.output.resolve()
    assert not out.is_relative_to(HERE),'Use a directory outside the released evidence'
    if args.data_root:os.environ['WEED_DATA_DIR']=str(args.data_root.resolve())
    out.mkdir(parents=True,exist_ok=True)
    context={'source_package':str(HERE),'benchmark':args.benchmark}
    marker=out/'REPRODUCTION_CONTEXT.json'
    if marker.exists():assert json.loads(marker.read_text())==context
    else:
        assert not list(out.iterdir()),'New output directory must be empty'
        marker.write_text(json.dumps(context,indent=2)+'\n')
    with hold_lock(out/'execution.lock'):
        sys.path.insert(0,str(CODE))
        if args.benchmark=='deeplab':
            import deeplab_benchmark as b
            b.OUT=out
            m,s=b.config()
            if not args.evaluate:b.train(m,s)
            b.evaluate(s);b.summarize()
        else:
            import source_inputs_benchmark as b
            b.OUT=out
            m,s=b.prepare()
            if not args.evaluate:b.spectral_train(m,s);b.forest_train(m,s)
            b.evaluate(s)


if __name__=='__main__':main()
