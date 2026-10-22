"""Export EgoTouch latent sequences from a trusted local checkpoint."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .data import RawEgoTouch
from .model import DinoBackbone, EgoTouchLatentModel, MODEL_ID, REVISION, load_vae_weights
from .train import features

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--split",default="val")
    p.add_argument("--limit",type=int,default=32); p.add_argument("--views",choices=("ego","all"),default="ego")
    p.add_argument("--output",default="processing/latents/egotouch_validation.npz"); args=p.parse_args()
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); cfg=state["config"]
    if cfg.get("latent_objective")!="vae": raise ValueError("This exporter supports image-only VAE checkpoints")
    if state["backbone"]!=MODEL_ID or state["backbone_revision"]!=REVISION: raise ValueError("Backbone mismatch")
    keys=("width","heads","temporal_layers","cross_view_layers","latent_dim","dropout","stochastic")
    backbone=DinoBackbone(); model=EgoTouchLatentModel(input_dim=backbone.dim,
        **{k:cfg[k] for k in keys})
    load_vae_weights(model,state["model"]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device); model.to(device).eval()
    ds=RawEgoTouch(cfg["root"],split=args.split,frames=cfg["frames"],stride=cfg["stride"],size=cfg["size"])
    ids=np.linspace(0,len(ds)-1,min(args.limit,len(ds)),dtype=int).tolist(); loader=DataLoader(Subset(ds,ids),batch_size=cfg["batch_size"])
    latents,predictions,episodes=[],[],[]
    for batch in loader:
        mask=batch["view_mask"].to(device)
        if args.views=="ego": mask[:,:,1:]=False
        with torch.autocast("cuda",torch.bfloat16,enabled=device.type=="cuda"):
            out=model(features(backbone,batch,device),mask)
        latents.append(out["mu"].float().cpu().numpy()); predictions.append(out["pressure"].float().cpu().numpy()); episodes.extend(batch["episode"])
    destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream,mu=np.concatenate(latents),predictions=np.concatenate(predictions),episode=np.array(episodes),window_index=np.array(ids),metadata=json.dumps({"checkpoint":args.checkpoint,"views":args.views,"backbone":MODEL_ID,"revision":REVISION}))
    print(f"Saved {len(episodes)} EgoTouch latent sequences to {destination}")

if __name__=="__main__": main()
