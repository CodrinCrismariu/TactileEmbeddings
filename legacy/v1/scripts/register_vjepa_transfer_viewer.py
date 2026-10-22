#!/usr/bin/env python3
"""Register completed V-JEPA probes for viewing. No fitting or evaluation sweep."""
import json
from pathlib import Path

import torch

from tactilelatent.aligned_frames import file_hash,write_json
from tactilelatent.train_alignment import atomic_checkpoint
from tactilelatent.viewer_transfer import load_transfer_head


def register(root,probe_name,checkpoint):
    folder=root/'probes'/probe_name
    if not (folder/'results.json').exists():return []
    summary=json.loads((folder/'results.json').read_text())
    encoder=Path(checkpoint['path']);digest=file_hash(encoder)
    if digest!=checkpoint['sha256']:raise ValueError('Evaluated encoder changed')
    state=torch.load(encoder,map_location='cpu',weights_only=False,mmap=True)
    if state['config']['heldout_source']!='ftp_motiontrans' or state['epoch']+1!=checkpoint['epoch']:
        raise ValueError('Wrong held-out encoder epoch')
    normalization=torch.load(folder/'normalization.pt',map_location='cpu',weights_only=False)
    result=[]
    for seed in summary['head_seeds']:
        original=folder/f'seed_{seed}'/'head.pt';metrics=original.with_name('results.json')
        saved=torch.load(original,map_location='cpu',weights_only=False)
        evaluated=json.loads(metrics.read_text())
        if (saved['seed']!=seed or saved['selected_epochs']!=evaluated['selected_head_epochs']
                or saved['input_dimension']!=state['model_config']['latent_dim']
                or not torch.equal(saved['mean'],normalization['mean'])
                or not torch.equal(saved['std'],normalization['std'])):
            raise ValueError('Head/normalization differs from completed evaluation')
        destination=Path(state['config']['output'])/'evaluations'/f'motiontrans_zscore_epoch{checkpoint["epoch"]:03d}_seed{seed}'
        packaged=dict(head=saved['head'],mean=saved['mean'],std=saved['std'],feature_normalization='fit_zscore',
            config=dict(checkpoint=str(encoder.resolve()),hidden_dims=summary['head'][1:-1]),
            checkpoint_sha256=digest,selected_epochs=saved['selected_epochs'],seed=seed,
            provenance=dict(original_head=str(original),original_head_sha256=file_hash(original),
                original_results=str(metrics),original_results_sha256=file_hash(metrics),
                evaluation_provenance_sha256=file_hash(root/'provenance.json')))
        # Verify the exported decoder matches the source probe before publishing.
        decoder=load_transfer_head(packaged,saved['input_dimension']).eval()
        from tactilelatent.frozen_probe import head
        baseline=head(saved['input_dimension'],6,hidden_dims=summary['head'][1:-1]).eval()
        baseline.load_state_dict(saved['head'])
        x=torch.randn(7,saved['input_dimension'],generator=torch.Generator().manual_seed(45))
        with torch.no_grad():torch.testing.assert_close(decoder(x),baseline((x-saved['mean'])/saved['std']),rtol=0,atol=0)
        destination.mkdir(parents=True,exist_ok=True)
        atomic_checkpoint(destination/'decoder.pt',packaged)
        write_json(destination/'results.json',dict(source='ftp_motiontrans',frozen_encoder=True,
            checkpoint_sha256=digest,checkpoint_epoch=state['epoch'],target_shape=[6,1,1],
            seed=seed,validation=evaluated['validation'],feature_normalization='fit_zscore',
            original_results=str(metrics),viewer_export_only=True))
        result.append(dict(path=str(destination),epoch=checkpoint['epoch'],seed=seed))
    return result


def main():
    torch.set_num_threads(4);published=[]
    best=Path('processing/evaluations/motiontrans_vjepa21_best_v1')
    p=json.loads((best/'provenance.json').read_text())
    for name,checkpoint in p['checkpoints'].items():published+=register(best,name,checkpoint)
    sweep=Path('processing/evaluations/motiontrans_vjepa21_deterministic_epochs_v1')
    p=json.loads((sweep/'provenance.json').read_text())
    for epoch,checkpoint in p['checkpoints'].items():published+=register(sweep,f'epoch_{int(epoch):03d}',checkpoint)
    write_json(best/'viewer_registration.json',dict(heads=published,training_started=False,evaluation_resumed=False))
    print(json.dumps(published,indent=2))


if __name__=='__main__':main()
