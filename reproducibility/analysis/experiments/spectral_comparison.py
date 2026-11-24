"""Repeat spectral configurations on the revised common source split."""
import run_revision as r
CONFIGS={'rgb':[0,1,2],'rgb_nir_re':[0,1,2,3,4],'rgb_indices':[0,1,2,5,6,7]}

def train():
    manifest=r.prepare();stats=r.source_stats(manifest['source']['training'])
    for norm in r.NORMS:
        for config,channels in CONFIGS.items():
            tr=r.Data(manifest['source']['training'],norm,stats,True)
            val=r.Data(manifest['source']['validation'],norm,stats)
            tr.images=[a[channels] for a in tr.images];val.images=[a[channels] for a in val.images]
            for seed in r.SOURCE_SEEDS:
                path=r.OUT/f'spectral_{norm}_{config}_seed{seed}'
                if (path/'complete.json').exists():continue
                r.legacy.seed_all(seed)
                r.train_one(path,tr,val,seed,'source',model=r.UNet(len(channels),3))
            del tr,val

if __name__=='__main__':
    raise SystemExit('This file provides model/training helpers. Use reproduce_target.py, staged_revision.py or the documented benchmark scripts; the superseded full-grid entry point is disabled.')
