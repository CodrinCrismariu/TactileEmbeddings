"""One-seed frozen DINO/random/trained-trunk readout comparison on MotionTrans."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader,Subset

from tactilelatent.aligned_frames import file_hash,manifest_signature,write_json
from tactilelatent.alignment_data import evaluation_dataset
from tactilelatent.alignment_model import MODEL_TYPE,JointSensorAlignment,model_from_state
from tactilelatent.frozen_probe import select_indices,seed_all
from tactilelatent.model import DinoBackbone,MODEL_ID,REVISION
from tactilelatent.representation_probe import pooled_dino,temporal_summary,fit_probe_sweep,paired_difference
from tactilelatent.train import features
from tactilelatent.train_alignment import model_weight_hash

NAMES=('dino_single_frame_768','dino_temporal_summary_2304',
       'reconstructed_epoch000_latent_32','epoch007_latent_32','epoch100_latent_32',
       'reconstructed_epoch000_trunk_768','epoch007_trunk_768','epoch100_trunk_768')


def atomic_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def extract(output,manifest,checkpoints,device,status):
    """A shared frozen DINO pass writes only pooled features, not patch banks."""
    checkpoint_hashes={name:file_hash(path) for name,path in checkpoints.items()}
    audit_path=output/'initialization_audit.json'
    audit=json.loads(audit_path.read_text())
    if not audit.get('verified_16d_hash') or not audit.get('original_32d_alignment_code_matches_archive') or not audit.get('trunk_equal_to_verified_16d_initialization'):
        raise ValueError('Historical initialization reconstruction did not pass its audit')
    reconstructed=output/'reconstructed_epoch000.pt'
    provenance=dict(manifest=manifest,manifest_sha256=file_hash(manifest),data=manifest_signature(manifest),
        checkpoints={k:dict(path=str(checkpoints[k]),sha256=v) for k,v in checkpoint_hashes.items()},
        backbone=MODEL_ID,backbone_revision=REVISION,source='ftp_motiontrans',seed=45,
        code={p:file_hash(p) for p in ('scripts/evaluate_representation_screen.py','src/tactilelatent/representation_probe.py',
                                      'src/tactilelatent/model.py','src/tactilelatent/alignment_data.py')},
        initialization_audit_sha256=file_hash(audit_path),reconstructed_epoch000_sha256=file_hash(reconstructed),
        pooling='Per-frame arithmetic mean over patch tokens and available camera slots; temporal DINO concatenates current frame, eight-frame clip mean, and current-minus-previous difference.',
        tuning=dict(head_seeds=[45,46,47],learning_rates=[.0003,.001,.003],partition_seed=45,standardization='fit-only z-score'),
        untrained_note='Reconstructed from archived original Gaussian constructor and seed-45 DINO→model initialization order. This path exactly reproduces the recorded 16D initialization hash and the shared trunk tensors match. No original 32D initialization checkpoint/hash exists for independent full-state verification.')
    pinned=output/'provenance.json'
    if pinned.exists() and json.loads(pinned.read_text())!=provenance:
        raise ValueError('Frozen representation provenance changed')
    write_json(pinned,provenance)
    states={}
    for name,path in checkpoints.items():
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        expected=7 if name=='epoch007' else 100
        if (state['epoch']+1!=expected or state['model_type']!=MODEL_TYPE or state['backbone_revision']!=REVISION
                or state['config']['heldout_source']!='ftp_motiontrans' or state['config'].get('train_sources') is not None
                or state['model_config']['latent_dim']!=32 or state['provenance']['data']!=provenance['data']):
            raise ValueError('Incorrect epoch or held-out checkpoint contract')
        states[name]=state
    if states['epoch007']['model_config']!=states['epoch100']['model_config']:
        raise ValueError('Trained encoder architectures differ')
    missing=any(not (output/'features'/name/f'{split}.pt').exists() for name in NAMES for split in ('train','val'))
    if not missing:
        for split in ('train','val'):
            common=torch.load(output/'features'/f'{split}_targets.pt',map_location='cpu',weights_only=False)
            for name in NAMES:
                cached=torch.load(output/'features'/name/f'{split}.pt',map_location='cpu',weights_only=False,mmap=True)
                if cached['provenance']!=provenance or cached['indices']!=common['indices']:
                    raise ValueError('Stale feature cache')
        return provenance
    seed_all(45)
    backbone=DinoBackbone().to(device).eval().requires_grad_(False)
    random_model=JointSensorAlignment(**states['epoch007']['model_config']).to(device).eval().requires_grad_(False)
    saved=torch.load(reconstructed,map_location='cpu',weights_only=False,mmap=True)
    if saved['trained_epochs']!=0 or saved['initialization_audit']!=audit or saved['model_config']!=states['epoch007']['model_config']:
        raise ValueError('Reconstructed initial checkpoint differs from its audit')
    random_model.load_state_dict(saved['model'],strict=True)
    if model_weight_hash(random_model)!=audit['reconstructed_32d_hash']:
        raise ValueError('Reconstructed initial weights changed')
    del saved
    write_json(output/'untrained_initialization.json',dict(seed=45,model_weight_sha256=model_weight_hash(random_model),
        note=provenance['untrained_note']))
    trained={name:model_from_state(state).to(device).eval().requires_grad_(False) for name,state in states.items()}
    del states
    for split in ('train','val'):
        dataset=evaluation_dataset(manifest,split=split,sources=['ftp_motiontrans'],include_targets=True)
        indices=select_indices(dataset)
        loader=DataLoader(Subset(dataset,indices),batch_size=8,num_workers=4,shuffle=False,pin_memory=device.type=='cuda')
        moments={name:[] for name in NAMES};targets=[];valid=[];episodes=[]
        for step,batch in enumerate(loader):
            if any(s!='ftp_motiontrans' for s in batch['source']):
                raise ValueError('Mixed source in target probe')
            with torch.no_grad(),torch.autocast(device.type,torch.bfloat16,enabled=device.type=='cuda'):
                encoded=features(backbone,batch,device);mask=batch['view_mask'].to(device)
                dino=pooled_dino(encoded,mask)
                initial_context=random_model.visual.encode_context(encoded,mask)
                result=dict(dino_single_frame_768=dino,dino_temporal_summary_2304=temporal_summary(dino),
                    reconstructed_epoch000_trunk_768=initial_context,
                    reconstructed_epoch000_latent_32=random_model.visual.prior_mu(initial_context))
                for name,model in trained.items():
                    context=model.visual.encode_context(encoded,mask)
                    result[f'{name}_trunk_768']=context
                    result[f'{name}_latent_32']=model.visual.prior_mu(context)
            for name,value in result.items():
                moments[name].append(value.float().cpu())
            targets.append(batch['target'].cpu());valid.append(batch['target_mask'].cpu());episodes.extend(batch['episode_id'])
            if step%100==0:
                status('extracting_frozen_features',split=split,batch=step,batches=len(loader))
        common=dict(target=torch.cat(targets),valid=torch.cat(valid),episode_id=episodes,indices=indices,provenance=provenance)
        atomic_save(output/'features'/f'{split}_targets.pt',common)
        for name in NAMES:
            atomic_save(output/'features'/name/f'{split}.pt',dict(values=torch.cat(moments[name]),indices=indices,provenance=provenance))
        del common,moments,targets,valid
    del backbone,random_model,trained;gc.collect();torch.cuda.empty_cache()
    if any(file_hash(path)!=checkpoint_hashes[name] for name,path in checkpoints.items()):
        raise ValueError('Encoder checkpoint changed during extraction')
    return provenance


def summarize(output,results,provenance):
    comparisons={}
    reference=results.get('epoch007_latent_32')
    if reference:
        comparisons={name:paired_difference(reference,row) for name,row in results.items() if name!='epoch007_latent_32'}
    if 'epoch007_trunk_768' in results and 'epoch100_trunk_768' in results:
        comparisons['trunk100_vs_trunk007']=paired_difference(results['epoch007_trunk_768'],results['epoch100_trunk_768'])
    write_json(output/'summary.json',dict(results=results,comparisons=comparisons,provenance=provenance,
        uncertainty='Point-readout comparison only. No variance or calibrated predictive uncertainty is supplied.',
        interpretation='A probe matching another representation establishes no measured benefit under this protocol; it does not prove absence of transferable information. Three head seeds and three tune-selected LRs are used on a development target; these are not three representation-training seeds.'))
    lines=['# Frozen representation comparison on MotionTrans','',
        'All encoders and DINO are frozen. Each representation receives fit-only z-score normalization and the same three-hidden-layer 512-unit MLP. Three LRs are selected by mean tune loss over three head seeds; only then are heads refit and evaluated. All use deterministic means and a fixed episode fit/tune split. These values use a new matched epoch-7 baseline rather than the older Gaussian-sampled readout protocol.','',
        '| Representation | Input dimensions | Head parameters | Selected LR | Mean MAE ± head-seed SD | Episode-bootstrap 95% CI |','|---|---:|---:|---:|---:|---|']
    for name,row in results.items():
        lo,hi=row['validation']['episode_bootstrap_mae_ci95']
        lines.append(f"| {name} | {row['input_dimension']} | {row['head_parameters']:,} | {row['selected_learning_rate']:g} | {row['validation']['mae']:.6f} ± {row['validation']['mae_head_seed_std']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines+=['','Paired differences against the matched epoch-7 32D latent (negative favors the candidate):','',
        '| Candidate | MAE difference | Paired episode-bootstrap 95% CI |','|---|---:|---|']
    for name,row in comparisons.items():
        lo,hi=row['paired_episode_bootstrap_ci95'];lines.append(f"| {name} | {row['candidate_minus_reference_mae']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','DINO is mean-pooled over patch tokens and all valid cameras. Its temporal variant concatenates the current frame, eight-frame clip mean, and within-clip frame difference (first difference zero). Trunk features are exactly the vectors feeding the Gaussian heads: mean spatial pooling followed by the learned image_pool transformation. Learned camera attention remains part of the trunk computation; DINO does not borrow trained camera weights. No tactile channels or frames are pooled or resized.',
        '', 'Wider heads have larger first layers despite identical hidden depth and width; parameter counts are reported. The fit-only z-score remains fixed during refit. Bootstrap intervals resample episodes after averaging errors over the three heads; head-fit MAE standard deviation is reported separately.',
        '', provenance['untrained_note'],
        '', 'If DINO or the random trunk matches epoch 7, this shows no measured benefit from tactile representation training under this readout and target split. It cannot establish that tactile training added zero transferable information. If wide trunk features preserve transfer while the 32D latent degrades, the bottleneck/projection is implicated; if wide features also degrade, the change extends into the trunk. A single readout configuration can also be sensitive to input scaling and optimization.',
        '', 'The target validation split has informed experimental design and is a development benchmark. All feature/target caches retain canonical clip order, native targets, episode IDs and immutable provenance.']
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('processing/evaluations/motiontrans_representation_screen_v1'))
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    def status(stage,**details):
        row=dict(stage=stage,controller_pid=os.getpid(),elapsed_seconds=time.monotonic()-began,**details)
        write_json(output/'status.json',row);print(json.dumps(row),flush=True)
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parent=Path('processing/runs/joint_alignment_holdout_motiontrans')
    manifest='processing/joint_alignment/manifest.json'
    checkpoints=dict(epoch007=parent/'viewer_epochs/epoch_007.pt',epoch100=parent/'epoch_100.pt')
    try:
        provenance=extract(output,manifest,checkpoints,device,status)
        results={}
        for name in NAMES:
            folder=output/'probes'/name;path=folder/'results.json'
            if path.exists():
                saved=json.loads(path.read_text())
                if saved.get('provenance')!=provenance:
                    raise ValueError('Completed probe provenance changed')
                results[name]=saved
            else:
                data={}
                for split in ('train','val'):
                    common=torch.load(output/'features'/f'{split}_targets.pt',map_location='cpu',weights_only=False)
                    cached=torch.load(output/'features'/name/f'{split}.pt',map_location='cpu',weights_only=False)
                    if common['provenance']!=provenance or cached['provenance']!=provenance or cached['indices']!=common['indices']:
                        raise ValueError('Feature/target indices or provenance differ')
                    data[split]={**common,'values':cached['values']}
                status('fitting_frozen_readout',representation=name,completed=len(results),total=len(NAMES))
                results[name]=fit_probe_sweep(data['train'],data['val'],folder,device=device,
                    progress=lambda stage,**details:status(stage,representation=name,**details))
                results[name]['provenance']=provenance
                write_json(path,results[name]);del data;gc.collect();torch.cuda.empty_cache()
            summarize(output,results,provenance)
        status('complete',summary=str(output/'summary.json'))
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':
    main()
