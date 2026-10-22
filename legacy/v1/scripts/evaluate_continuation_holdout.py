"""Full MotionTrans probes with shared DINO extraction across continuation arms."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from evaluate_invariance_holdout import paired_difference
from tactilelatent.aligned_frames import manifest_signature, write_json
from tactilelatent.alignment_data import evaluation_dataset
from tactilelatent.alignment_model import model_from_state
from tactilelatent.frozen_probe import select_indices, sha256
from tactilelatent.model import DinoBackbone, MODEL_ID, REVISION
from tactilelatent.paired_probe_cache import extract_split

RUNS = ('control', 'source_stopping', 'retention', 'combined')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('processing/runs/continuation_32d_v1'))
    parser.add_argument('--epoch', type=int, required=True)
    args = parser.parse_args()
    root, epoch = args.root, args.epoch
    names = ['control'] if epoch == 7 else list(RUNS)
    output = root/'evaluations'/f'epoch_{epoch:03d}'
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    def status(stage, **details):
        row = dict(stage=stage, epoch=epoch, controller_pid=os.getpid(), elapsed_seconds=time.monotonic()-began, **details)
        write_json(output/'status.json', row); print(json.dumps(row), flush=True)
    try:
        torch.set_num_threads(4); device=torch.device('cuda')
        models, snapshots, digests = {}, {}, {}
        manifest, signature = None, None
        for name in names:
            checkpoint = root/name/'viewer_epochs'/f'epoch_{epoch:03d}.pt'
            snapshot = root/name/'evaluations'/f'motiontrans_epoch_{epoch:03d}'/'checkpoint.pt'
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                os.link(checkpoint, snapshot)
            if sha256(snapshot) != sha256(checkpoint):
                raise ValueError('Immutable probe checkpoint changed')
            state = torch.load(snapshot, map_location='cpu', weights_only=False, mmap=True)
            if (state['epoch']+1 != epoch or state['config']['heldout_source'] != 'ftp_motiontrans'
                    or state['backbone'] != MODEL_ID or state['backbone_revision'] != REVISION
                    or state['model_config']['latent_dim'] != 32 or state['model_config'].get('latent_mode', 'gaussian') != 'gaussian'):
                raise ValueError('Incorrect epoch, held-out protocol, or architecture')
            current_manifest = state['config']['manifest']
            if state['provenance']['data'] != manifest_signature(current_manifest):
                raise ValueError('Canonical data provenance changed')
            if manifest is not None and (manifest != current_manifest or signature != state['provenance']['data']):
                raise ValueError('Probe data contracts differ')
            manifest, signature = current_manifest, state['provenance']['data']
            models[name] = model_from_state(state).to(device).eval().requires_grad_(False)
            snapshots[name], digests[name] = snapshot, sha256(snapshot)
            del state
        # Baseline epoch-7 means/logvars reuse the exact original cache if present.
        missing = any(not (s.parent/'features'/f'{split}_features.pt').exists() for s in snapshots.values() for split in ('train','val'))
        if missing:
            backbone = DinoBackbone().to(device).eval().requires_grad_(False)
        else:
            backbone = None
        for split in ('train','val'):
            dataset = evaluation_dataset(manifest, split=split, sources=['ftp_motiontrans'])
            indices = select_indices(dataset)
            destinations = {name:snapshot.parent/'features'/f'{split}_features.pt' for name,snapshot in snapshots.items()}
            extract_split(backbone, models, manifest, 'ftp_motiontrans', split, indices, destinations,
                digests, sha256(manifest), device, batch_size=8, workers=4,
                progress=lambda split,batch,batches:status('extracting_features', split=split,batch=batch,batches=batches))
        del backbone, models, dataset; gc.collect(); torch.cuda.empty_cache()
        results = {}
        for name in names:
            report = f'continuation_probe_epoch_{epoch:03d}'
            command = [sys.executable,'-u','scripts/evaluate_alignment_checkpoint_sweep.py',
                '--run',str(root/name),'--epochs',str(epoch),'--report-name',report,'--heads','deeper',
                '--head-epochs','100','--patience','10','--samples','64','--seed','45','--workers','2',
                '--checkpoint-kind','viewer']
            with (output/f'{name}.log').open('a') as log:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status('fitting_head',run=name,worker_pid=process.pid)
                if process.wait():
                    raise RuntimeError(f'{name} transfer probe failed; see {output}/{name}.log')
            summary=json.loads((root/name/'evaluations'/report/'summary.json').read_text())
            if summary['completed_evaluations'] != 1 or summary['failures']:
                raise ValueError('Selected transfer head did not complete')
            results[name]=summary['results'][0]
            write_json(output/'partial_results.json',results)
        comparisons={}
        if epoch != 7:
            control=json.loads(Path(results['control']['results']).read_text())
            baseline=json.loads((root/'evaluations/epoch_007/summary.json').read_text())['results']['control']
            full_baseline=json.loads(Path(baseline['results']).read_text())
            for name,row in results.items():
                full=json.loads(Path(row['results']).read_text())
                comparison=paired_difference(control,full)
                comparison['candidate_minus_control_mae']=comparison.pop('relational_minus_control_mae')
                parent_comparison=paired_difference(full_baseline,full)
                parent_comparison['candidate_minus_parent_mae']=parent_comparison.pop('relational_minus_control_mae')
                comparisons[name]=dict(vs_control=comparison,vs_parent=parent_comparison,
                    change_vs_parent_percent=100*(row['deterministic_mae']/baseline['deterministic_mae']-1))
        write_json(output/'summary.json',dict(epoch=epoch,results=results,comparisons=comparisons,
            protocol='32→512→512→512→6; full MotionTrans episodes; same train episode tune split, seed 45, 100 maximum head epochs, patience 10, 64 MC draws; shared frozen DINO extraction; 5000 paired episode bootstrap draws.',
            note='Development benchmark informed experimental design; no target episodes enter representation updates. One seed, no hyperparameter screen. Existing uncertainty is uncalibrated.'))
        status('complete',summary=str(output/'summary.json'))
    except Exception as error:
        status('failed',error=str(error)); raise


if __name__ == '__main__':
    main()
