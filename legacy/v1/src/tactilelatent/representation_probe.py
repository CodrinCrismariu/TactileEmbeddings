"""Readout probes of frozen representations; no encoder/backbone optimization."""
from pathlib import Path

import numpy as np
import torch

from .evaluate_holdout_decoder import episode_diagnostics, regression_loss
from .frozen_probe import episode_partition, fit_head, head, seed_all


def pooled_dino(features, view_mask):
    """Mean over spatial patches and available cameras, separately per frame."""
    if features.ndim != 5 or view_mask.shape != features.shape[:3] or not view_mask.any(-1).all():
        raise ValueError('Expected B,T,V,P,D features and at least one camera per frame')
    values = features.float().mean(-2)
    return (values * view_mask[..., None]).sum(-2) / view_mask.sum(-1, keepdim=True)


def temporal_summary(values):
    """Current frame, mean of all eight clip frames, current-minus-previous.

    The first frame's difference is zero; no observations outside the clip
    are read. Every output remains paired to its original target frame.
    """
    if values.ndim!=3 or not torch.isfinite(values).all():
        raise ValueError('Expected finite B,T,D pooled frame features')
    difference=torch.zeros_like(values)
    difference[:,1:]=values[:,1:]-values[:,:-1]
    mean=values.mean(1,keepdim=True).expand_as(values)
    return torch.cat((values,mean,difference),dim=-1)


def check_pair(train, validation):
    for cache in (train, validation):
        x, y, valid = cache['values'], cache['target'], cache['valid']
        if (x.ndim != 3 or y.shape[:2] != x.shape[:2] or valid.shape != y.shape
                or len(cache['episode_id']) != len(x) or not torch.isfinite(x).all()
                or not torch.isfinite(y).all() or not valid.all()):
            raise ValueError('Nonfinite, misaligned or missing representation/target values')
    if train['values'].shape[-1] != validation['values'].shape[-1]:
        raise ValueError('Feature dimensions differ across splits')
    if set(train['episode_id']) & set(validation['episode_id']):
        raise ValueError('Episode leakage between training and validation')


def feature_table(cache, ids, device):
    values = cache['values'][ids].flatten(0, 1).detach().float().to(device)
    # Existing fit_head's two training terms now see identical frozen means.
    # Every representation, including trained Gaussian latents, uses this rule.
    return dict(mu=values, logvar=torch.full_like(values, -torch.inf),
                y=cache['target'][ids].flatten(0, 1).flatten(1).detach().float().to(device))


@torch.no_grad()
def score(network, train, validation, device, seed):
    network.eval()
    values=validation['values'].flatten(0,1).float().to(device)
    target=validation['target'].flatten(2).float().to(device)
    point=torch.cat([network(x) for x in values.split(512)]).reshape_as(target)
    baseline=train['target'].float().mean((0,1)).flatten().to(device)
    error=(point-target).abs()
    return dict(mae=error.double().mean().item(), mse=(point-target).square().double().mean().item(),
        per_channel_mae=error.double().mean((0,1)).cpu().tolist(),
        mean_target_baseline_mae=(target-baseline).abs().double().mean().item(),
        **episode_diagnostics(point,target,validation['episode_id'],seed))


def fit_probe(train, validation, output, seed=45, epochs=100, patience=10, device=None):
    check_pair(train,validation)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(seed)
    fit_ids=[i for i,e in enumerate(train['episode_id']) if episode_partition(e,seed)=='fit']
    tune_ids=[i for i,e in enumerate(train['episode_id']) if episode_partition(e,seed)=='tune']
    if not fit_ids or not tune_ids:
        raise ValueError('Missing episode-level fit/tune partition')
    dimension=train['values'].shape[-1]
    fit,tune=feature_table(train,fit_ids,device),feature_table(train,tune_ids,device)
    network=head(dimension,6,hidden_dims=[512,512,512]).to(device)
    selection=fit_head(network,fit,tune,regression_loss,epochs=epochs,patience=patience,seed=seed,device=device)
    del fit,tune,network
    seed_all(seed)
    network=head(dimension,6,hidden_dims=[512,512,512]).to(device)
    refit=fit_head(network,feature_table(train,range(len(train['values'])),device),None,
        regression_loss,fixed_epochs=selection['selected_epochs'],seed=seed,device=device)
    metrics=score(network,train,validation,device,seed+2026)
    result=dict(input_dimension=dimension,head=[dimension,512,512,512,6],seed=seed,
        head_parameters=sum(p.numel() for p in network.parameters()),selected_head_epochs=selection['selected_epochs'],
        train_clips=len(train['values']),fit_clips=len(fit_ids),tune_clips=len(tune_ids),validation_clips=len(validation['values']),
        feature_statistics=dict(mean=float(train['values'].detach().float().mean()),std=float(train['values'].detach().float().std())),
        protocol='Frozen raw features; same three hidden layers, AdamW/loss/dropout/tune-refit protocol; means only for ALL representations. No learned representation updates, feature normalization, random latent draws, or uncertainty claim.',
        validation=metrics,selection=selection,refit=refit)
    torch.save(dict(head=network.cpu().state_dict(),selected_epochs=selection['selected_epochs'],input_dimension=dimension),output/'head.pt')
    return result


def paired_difference(reference,candidate,seed=45):
    a={r['episode_id']:r for r in reference['validation']['episode_errors']}
    b={r['episode_id']:r for r in candidate['validation']['episode_errors']}
    if set(a)!=set(b) or any(a[e]['count']!=b[e]['count'] for e in a):
        raise ValueError('Paired episode errors are not comparable')
    episodes=sorted(a)
    difference=np.array([b[e]['absolute_sum']-a[e]['absolute_sum'] for e in episodes])
    counts=np.array([a[e]['count'] for e in episodes])
    draws=np.random.default_rng(seed).integers(len(episodes),size=(5000,len(episodes)))
    sampled=difference[draws].sum(1)/counts[draws].sum(1)
    return dict(candidate_minus_reference_mae=float(difference.sum()/counts.sum()),
        paired_episode_bootstrap_ci95=np.quantile(sampled,[.025,.975]).tolist(),episodes=len(episodes),bootstrap_replicates=5000)


def fit_statistics(train,fit_ids):
    """Z-score statistics use only fit episodes, including at final refit."""
    values=train['values'][fit_ids].detach().float().flatten(0,1)
    mean=values.mean(0)
    std=values.std(0,unbiased=False).clamp_min(1e-6)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError('Invalid fit-only feature statistics')
    return mean,std


def aggregate_heads(rows,seed=2071):
    """Mean readout errors over head seeds; bootstrap episodes separately."""
    maes=np.array([r['validation']['mae'] for r in rows])
    reference={r['episode_id']:r for r in rows[0]['validation']['episode_errors']}
    episode_rows=[]
    for episode in sorted(reference):
        paired=[]
        for row in rows:
            current={r['episode_id']:r for r in row['validation']['episode_errors']}
            if set(current)!=set(reference) or current[episode]['count']!=reference[episode]['count']:
                raise ValueError('Head seeds have different validation episodes')
            paired.append(current[episode])
        episode_rows.append(dict(episode_id=episode,count=reference[episode]['count'],
            absolute_sum=float(np.mean([r['absolute_sum'] for r in paired])),
            per_channel_mae=np.mean([r['per_channel_mae'] for r in paired],axis=0).tolist()))
    totals=np.array([r['absolute_sum'] for r in episode_rows]);counts=np.array([r['count'] for r in episode_rows])
    draws=np.random.default_rng(seed).integers(len(episode_rows),size=(5000,len(episode_rows)))
    sampled=totals[draws].sum(1)/counts[draws].sum(1)
    return dict(mae=float(maes.mean()),mae_head_seed_std=float(maes.std(ddof=1)) if len(maes)>1 else 0.,
        mae_head_seed_min=float(maes.min()),mae_head_seed_max=float(maes.max()),
        mse=float(np.mean([r['validation']['mse'] for r in rows])),
        per_channel_mae=np.mean([r['validation']['per_channel_mae'] for r in rows],axis=0).tolist(),
        within_episode_channel_correlations=np.mean([r['validation']['within_episode_channel_correlations'] for r in rows],axis=0).tolist(),
        episode_errors=episode_rows,episode_bootstrap_mae_ci95=np.quantile(sampled,[.025,.975]).tolist(),
        episodes=len(episode_rows),bootstrap_replicates=5000,
        bootstrap_note='Resampled episodes after averaging each episode absolute error over the three fitted heads. Head-fit standard deviation is reported separately; these are not three representation-training seeds.')


def fit_probe_sweep(train,validation,output,seeds=(45,46,47),learning_rates=(.0003,.001,.003),
                    epochs=100,patience=10,device=None,progress=None):
    """Three head seeds × three LRs; shared LR selected by mean tune loss.

    Validation labels are used only after all nine tuning trials have chosen
    the LR. Refit each seed for its selected duration on all training episodes.
    """
    import json
    from .aligned_frames import write_json
    check_pair(train,validation)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fit_ids=[i for i,e in enumerate(train['episode_id']) if episode_partition(e,45)=='fit']
    tune_ids=[i for i,e in enumerate(train['episode_id']) if episode_partition(e,45)=='tune']
    if not fit_ids or not tune_ids:
        raise ValueError('Missing fixed seed-45 episode partition')
    mean,std=fit_statistics(train,fit_ids)
    torch.save(dict(mean=mean,std=std,fit_episode_ids=sorted({train['episode_id'][i] for i in fit_ids}),
        partition_seed=45,note='Fit-only statistics retained for tune, validation, and all-training refit.'),output/'normalization.pt')
    def normalized(cache):
        return {**cache,'values':(cache['values'].detach().float()-mean)/std}
    train=normalized(train);validation=normalized(validation)
    dimension=train['values'].shape[-1]
    fit,tune=feature_table(train,fit_ids,device),feature_table(train,tune_ids,device)
    signature=dict(seeds=list(seeds),learning_rates=list(learning_rates),epochs=epochs,patience=patience,
        partition_seed=45,fit_clips=len(fit_ids),tune_clips=len(tune_ids),input_dimension=dimension,
        weight_decay=.0001,hidden_dims=[512,512,512],normalization='fit-only per-coordinate z-score; floor 1e-6')
    trials=[]
    for rate in learning_rates:
        for seed in seeds:
            path=output/'tuning_trials'/f'lr_{rate:g}_seed_{seed}.json'
            if path.exists():
                result=json.loads(path.read_text())
                if result['signature']!=signature:
                    raise ValueError('Readout tuning configuration changed')
            else:
                if progress:progress('tuning_head',seed=seed,learning_rate=rate)
                seed_all(seed)
                network=head(dimension,6,hidden_dims=[512,512,512]).to(device)
                selection=fit_head(network,fit,tune,regression_loss,epochs=epochs,patience=patience,
                    seed=seed,device=device,lr=rate,weight_decay=.0001)
                result=dict(seed=seed,learning_rate=rate,selection=selection,signature=signature)
                write_json(path,result);del network
            trials.append(result)
    rate_losses={rate:float(np.mean([r['selection']['best_tune_loss'] for r in trials if r['learning_rate']==rate])) for rate in learning_rates}
    chosen=min(learning_rates,key=lambda rate:rate_losses[rate])
    write_json(output/'tuning_selection.json',dict(selected_learning_rate=chosen,mean_best_tune_loss=rate_losses,
        selection='Minimum mean best tune loss across three head seeds; no validation metrics used.',signature=signature))
    del fit,tune
    rows=[]
    for seed in seeds:
        selected=next(r for r in trials if r['seed']==seed and r['learning_rate']==chosen)['selection']['selected_epochs']
        folder=output/f'seed_{seed}';path=folder/'results.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['signature']!=signature or result['learning_rate']!=chosen or result['selected_head_epochs']!=selected:
                raise ValueError('Refitted head differs from selected training-only configuration')
        else:
            if progress:progress('refitting_head',seed=seed,learning_rate=chosen,selected_epochs=selected)
            seed_all(seed)
            network=head(dimension,6,hidden_dims=[512,512,512]).to(device)
            refit=fit_head(network,feature_table(train,range(len(train['values'])),device),None,
                regression_loss,fixed_epochs=selected,seed=seed,device=device,lr=chosen,weight_decay=.0001)
            result=dict(seed=seed,learning_rate=chosen,selected_head_epochs=selected,validation=score(network,train,validation,device,seed+2026),
                signature=signature,refit=refit)
            folder.mkdir(parents=True,exist_ok=True)
            torch.save(dict(head=network.cpu().state_dict(),input_dimension=dimension,mean=mean,std=std,
                selected_epochs=selected,seed=seed,learning_rate=chosen),folder/'head.pt')
            write_json(path,result);del network
        rows.append(result)
    return dict(schema_version=2,input_dimension=dimension,head=[dimension,512,512,512,6],
        head_parameters=(dimension+1)*512+2*(512+1)*512+(512+1)*6,
        selected_head_epochs=[r['selected_head_epochs'] for r in rows],selected_learning_rate=chosen,
        head_seeds=list(seeds),learning_rates=list(learning_rates),train_clips=len(train['values']),fit_clips=len(fit_ids),
        tune_clips=len(tune_ids),validation_clips=len(validation['values']),
        protocol='Frozen means only; fit-only z-score; fixed episode partition seed 45; three learning rates selected using mean tune loss over three head seeds; refit three heads; no validation model selection.',
        normalization='normalization.pt',tuning_trials=trials,seed_results=rows,validation=aggregate_heads(rows))
