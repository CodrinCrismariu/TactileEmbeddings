# Longer mirror autoencoder training

Continue **Baihu, EgoTouch, and Unit Bimanual** from immutable copies of the
best existing mirror-adapted weights. The short v2 preparation service is stopped
and disabled; its saved checkpoints, accepted Unit codes, and all v1 artifacts
remain unchanged. OpenGraph is still excluded until its taxel mapping is
verified. RDP Bimanual uses direct scalar targets and needs no autoencoder.

## Training

- One seed, 45; unchanged teacher architectures and sensor-code dimensions:
  Baihu 32D, EgoTouch 256D, Unit Bimanual 16D.
- Up to **100 additional epochs per source**, running sequentially on one GPU.
  At least 20 epochs; early stopping after 10 epochs without an improvement
  once a qualifying parity checkpoint has been found.
- AdamW at 5e-5, cosine decay to 1e-6, weight decay 1e-4, gradient clipping 1.
- 100,000 episode-balanced native training frames; refresh the sampled pool
  every five epochs. Shuffle frames across episodes to avoid long runs from
  one episode. Keep 10,000 fixed validation frames, episode-disjoint from fitting.
- Every update includes **both orientations of the same frames**. Retain each
  source's reconstruction loss. Combine orientation losses as
  `0.5*(L_original+L_mirror)+0.25*abs(L_original-L_mirror)`.
  This puts 75% gradient weight on the worse side and 25% on the better side;
  worsening either side never improves this objective.
- No RGB preprocessing/caching, spatial downsampling, extra heads, extra seeds,
  or production joint-training run.

## Selection and success criteria

Re-evaluate the original unaugmented teacher and continuation parent using the
same fixed validation frames and execution path. Preserve original overall and
contact/active-motion MAE within **10% of the original unaugmented teacher**.
Require both orientations to beat their zero baselines. The worse orientation
must not regress against the continuation parent, including active-motion error.
These guards also apply separately to each Baihu platform.

The parity target is `max(error_original,error_mirror)/min(error_original,error_mirror) <= 1.10`
for overall and contact/active-motion MAE, and for each Baihu platform. Among
quality-qualified candidates, prioritize checkpoints meeting parity, then select
the lowest normalized worst-orientation error. Similarity obtained by degrading
the original beyond the quality limit cannot qualify.

Improved, quality-qualified checkpoints appear in the viewer while training.
Only checkpoints that also meet the parity target receive promoted paired-code
caches. If a source misses the target by the epoch limit, report that explicitly;
keep prior production artifacts usable and do not silently promote it.

## Artifacts and operation

Config: `configs/mirror_balanced_training_v3.json`.
Command: `.venv/bin/python -u scripts/train_balanced_mirror_autoencoders.py`.
Persistent service: `tactilelatent-bimanual-mirror-v3.service`.

All new artifacts live under `processing/bimanual_mirror_v3/`:
immutable parent checkpoints, per-source fixed baselines, versioned best
checkpoints, resumable `last.pt`, metrics, teacher quality receipts, paired code
caches, `ready.json`, `status.json`, and continuous `progress.jsonl`.
Both orientations use the **same adapted encoder and decoder**. Accepted
checkpoints are immutable and individually hashed. Native target hashes,
sampling, optimizer/RNG state, and implementation hashes are recorded.

After all three sources pass, cache both target orientations through each
accepted encoder, refit equal-orientation train-only code statistics, and run
the existing real V-JEPA/SAM and paired reconstruction checks. Publish a separate
joint configuration without starting it. The viewer retains all historical
autoencoders and shows the continuation epoch and parity status.
