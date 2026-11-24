"""Explicitly rerun one bounded target stage in a separate output directory."""
import argparse
from execution_lock import hold_lock
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
CODE=HERE/'analysis/experiments'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',type=int,choices=[1,2,3,4],required=True)
    ap.add_argument('--output',type=Path,required=True,help='Separate directory for fresh or resumed results')
    ap.add_argument('--data-root',type=Path,help='Directory containing the patches subdirectory')
    ap.add_argument('--evaluate',action='store_true',help='Evaluate a completed stage; does not train')
    ap.add_argument('--summarize',action='store_true',help='Summarize completed test results; does not train')
    args=ap.parse_args()
    assert not(args.evaluate and args.summarize)
    output=args.output.resolve()
    assert not output.is_relative_to(HERE),'Use a directory outside the released evidence package'
    if args.data_root:os.environ['WEED_DATA_DIR']=str(args.data_root.resolve())
    output.mkdir(parents=True,exist_ok=True)
    marker=output/'REPRODUCTION_CONTEXT.json'
    context={'source_package':str(HERE),'scope':'Bounded target stages with released fixed source checkpoints and frozen partition design.'}
    if marker.exists():assert json.loads(marker.read_text())==context
    else:
        assert not list(output.iterdir()),'New output directory must be empty'
        marker.write_text(json.dumps(context,indent=2)+'\n')
    with hold_lock(output/'execution.lock'):
        sys.path.insert(0,str(CODE))
        import staged_revision as staged
        staged.BASE=output
        if args.stage==3:
            original=CODE/'staged_v2/source_source_fixed_seed42'
            assert (original/'best.pth').exists(),'Install weights_source_initializations.zip first'
            dest=output/'source_source_fixed_seed42'
            dest.mkdir(exist_ok=True)
            for name in ['best.pth','complete.json','history.json']:
                if not (dest/name).exists():shutil.copyfile(original/name,dest/name)
        if args.summarize:
            if args.stage==1:
                import summarize_stage1 as summary
                summary.ROOT=output;summary.FOLDER=output/'stage1';summary.main()
            else:
                import summarize_stage as summary
                summary.BASE=output;summary.main(args.stage)
        else:
            assert (CODE/'experiments_v1/source_patch_valid_seed42/best.pth').exists(),'Install weights_source_initializations.zip first'
            if args.evaluate:staged.evaluate(args.stage)
            else:staged.run(args.stage)


if __name__=='__main__':main()
