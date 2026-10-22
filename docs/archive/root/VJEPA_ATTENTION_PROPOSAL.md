> Historical record, relocated on October 21, 2026. Original path: `VJEPA_ATTENTION_PROPOSAL.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# V-JEPA attention readout for per-frame tactile latents

Proposal and evidence review, October 15, 2026. This document proposes a new task architecture; it does not launch training or resume the stopped MotionTrans evaluation.

## Recommendation

Keep frozen V-JEPA 2.1 Base and the existing whole-clip feature cache. Replace camera averaging plus the linear latent projection with **two frame-query attention decoder blocks and a nonlinear bottleneck MLP**. Produce exactly one 32D latent for each of the eight input images. Keep sensor-specific adapters downstream of this latent.

This is a hypothesis about a better task readout, not a guarantee of sensor invariance or improved transfer. The project has already shown that better source reconstruction can accompany worse transfer.

## What the implementation and results establish

The current path in `src/tactilelatent/vjepa_model.py` is:

```text
8 frames per available camera
→ frozen V-JEPA: 4 temporal groups × 576 patches × 768 channels
→ spatial arithmetic mean, cached separately for each camera
→ masked camera arithmetic mean
→ Linear(768, 64), split into two ordered 32D latents per group
→ source adapter or numeric head
```

V-JEPA processes the whole eight-frame clip. A temporal group is not an independent two-frame forward pass. However, there is no independently cached image descriptor for each member of a pair: fine distinctions between the two frames must be extracted from the contextualized tubelet features.

The source-only diagnostic is saved in `processing/diagnostics/vjepa_learning_v1/results.json`; its implementation is `scripts/diagnose_vjepa_learning.py`.

- Gradients are finite and nonzero; model weights changed substantially. There is no evidence that a detached projection or missing optimizer update explains the results.
- Current V-JEPA task parameters total 442,339, versus 44,309,348 for the previous DINO task model. Both counts exclude their frozen backbones. We removed most trainable capacity when changing backbones.
- For numeric sources, the current linear latent projection followed by a linear sigmoid head is effectively logistic linear regression on pooled frozen features. There is no learned nonlinear shared trunk in that path.
- On the fixed EgoTouch validation subset, predicted adjacent-frame code variation has RMS 0.0284 versus 0.3551 for the target. This is a temporal under-response measurement, not proof of its cause.
- On RDP, a diagnostic ridge fit on raw pooled 768D features reaches normalized native MAE 0.0292, versus 0.0557 for the current model. On bimanual RDP the corresponding values are 0.0269 versus 0.0754. This suggests useful information exists before the 32D readout. The ridge fit uses only 256 training clips, fixed regularization, and is not an optimized benchmark; it does not improve every source.
- In a repeated four-clip EgoTouch fit using copied epoch-30 weights, 200 SAM updates reduce loss from 1.2505 to 0.5081 at LR 5e-5, versus 0.0340 at LR 1e-3. This establishes optimization sensitivity and fitting ability, not validation improvement. Production weights were unchanged.
- The completed source-selected epoch-28 V-JEPA deterministic model transfers to MotionTrans at MAE 0.040215, compared with 0.043769 for the matched DINO epoch-7 32D representation. V-JEPA is not useless: its source fit is weaker while this transfer result is better. Frozen temporal DINO features remain stronger at 0.031462, with the important difference that they are 2304D.

Sources: `VJEPA_32D_EXPERIMENT.md`, `FROZEN_REPRESENTATION_SCREEN.md`, and their linked evaluation artifacts. Architecture, optimizer schedule, and capacity differ between the historical DINO and V-JEPA runs; these comparisons do not isolate a backbone effect.

## Data constraints that determine the design

There are eleven source datasets, with one to three available scene cameras. Some datasets have only four cached temporal tokens total; three-camera datasets have twelve. MotionTrans has one camera and remains excluded from representation training, normalization statistics, and source validation.

Targets range from one/two numeric values to pressure arrays and full-resolution optical tactile images. Nine sources use frozen AE code supervision; the two numeric sources use direct prediction. The shared architecture should not change width or queries based on the target sensor. Sensor identity is used only to select the adapter after the 32D latent.

Training diversity differs substantially: the manifest contains 47 bimanual RDP training episodes, 68 OpenGraph episodes, 134 RDP episodes, and thousands for several other sources. Their validation partitions can be as small as three, four, and six episodes. More sampled clips do not necessarily mean more independent interactions. These counts do not establish whether total data are sufficient, but they explain why per-source validation and episode-level uncertainty matter.

The cache stores float32 `[camera_clips, 4, 768]` features. It preserves camera separation and complete-clip context. It does **not** store the 24×24 spatial patch grid. The proposed first version uses exactly this cache and adds no visual extraction pass.

Camera-role metadata must be explicit. In particular, FreeTacMan's audited scene camera occupies slot 1, whereas slot 1 commonly stores a left camera in other sources. Keep physical file/slot mappings unchanged and attach verified semantic role metadata separately. Use a shared vocabulary such as scene/main, left camera, right camera, and unknown; do not infer physical calibration or exact viewpoint equivalence from these coarse roles. No dataset-specific embedding is introduced.

All tactile shapes, sensor-channel ordering, eight frame indices, validity masks, and teacher provenance remain unchanged. No tactile output is resized or averaged.

## Exact proposed architecture

### 1. Make a camera-separated memory

For each cached feature vector:

```text
LayerNorm(768) → Linear(768, 256)
  + verified camera-role embedding
  + temporal-position encoding
```

Flatten only the temporal and camera axes into 4–12 valid memory tokens. Retain masks for absent/dropped cameras. Zero-filled camera slots must never become valid attention keys. Position encodings describe positions within the clip, not physical elapsed time across datasets with different sampling rates.

### 2. Initialize eight content-dependent frame queries

For frame t, initialize its 256D query with the masked mean of projected camera features from temporal group floor(t/2), plus:

- a frame-position encoding identifying t within the eight-frame clip;
- a learned within-pair embedding identifying the first or second frame.

The initial local mean is an anchor, not the only information path: the camera-separated memory remains available to every attention layer. Two frames in the same tubelet receive distinct queries but share a single set of decoder weights. No tactile values enter the queries.

### 3. Apply two trainable decoder blocks

Each block uses pre-LayerNorm, residual connections, width 256, four attention heads, dropout 0.1, and:

1. Self-attention among the eight frame queries.
2. Cross-attention from those eight queries to all valid camera/temporal memory tokens.
3. A feed-forward MLP `256 → 1024 → 256`, with GELU.

Self-attention lets frame representations exchange context. Cross-attention learns which camera and temporal evidence each frame needs. The feed-forward networks learn nonlinear tactile-relevant combinations of frozen features. Both attention paths are bidirectional, matching the existing whole-clip model; this is not a causal online predictor.

V-JEPA already performs expensive video encoding. The small decoder learns the downstream readout rather than rebuilding a large patch transformer.

### 4. Form one 32D latent per image

Apply the same bottleneck network to each of the eight query outputs:

```text
LayerNorm(256) → Linear(256, 128) → GELU → Linear(128, 32)
```

Output: `[batch, 8, 32]`. There is no clip-level 32D vector shared by all frames, no concatenation of two frame latents for a sensor head, and no wide-feature bypass around the bottleneck.

For a later matched stochastic variant, share the 128D hidden activation and add a separate `Linear(128,32)` log-variance branch. Mean and variance both correspond to one image. Learned latent variance alone is not a calibrated predictive uncertainty estimate.

### 5. Keep the existing sensor-specific output path

```text
32D frame latent
├─ AE source: 32 → 128 → 128 → existing sensor-code dimension
│             → frozen decoder for native reconstruction evaluation
└─ numeric source: Linear(32, number_of_values) → sigmoid
```

The numeric head can remain simple because its shared input is now produced by a nonlinear attention network. MotionTrans transfer fits a new head to frozen 32D features using the existing protocol.

The input projection, two standard decoder blocks, and bottleneck contain 2,342,816 parameters before small role/query embeddings. Existing adapters add 393,123, giving approximately **2.74 million trainable parameters** total for the deterministic model. This count was checked with PyTorch modules; runtime has not been benchmarked.

## What this can and cannot accomplish

This architecture directly addresses equal camera averaging, the linear shared readout, and lack of task-trained temporal selection. Residual connections retain local pair information while attention can add relevant context. It does not guarantee that the 32D representation will be sensor-invariant: camera appearance and availability can still reveal the dataset, and source-code losses can still favor specialization.

Use shared weights and shared queries across all sources, valid-camera dropout, and existing per-source supervision. Do not add a dataset embedding, static episode context, a private latent bypass, FGW, or an additional temporal regularizer to the first architecture comparison. This keeps the result interpretable; none of those additions has been established as a reliable transfer improvement here.

Monitor native reconstruction and mean-code baselines, within-episode dynamics, and transfer. Effective rank and dataset-classifier accuracy remain diagnostics, not optimization targets or success criteria. Log each source's actual contribution: the current cap of 8,000 clips/source is not equal source weighting when smaller sources have fewer clips. Change sampling/weights only in a separate comparison.

The cache limitation remains: attention over one globally pooled vector cannot recover spatial patch positions. It may access local information implicitly encoded in those vectors, but cannot explicitly select a contact region. Likewise, distinct queries do not create missing frame-specific evidence if tubelet compression discarded it.

If spatial detail remains the limiting factor, a later version can use the same eight queries to cross-attend to the original spatial tokens, with spatial and camera position metadata. Query-to-token attention avoids a new full quadratic self-attention operation over every patch. That requires rerunning V-JEPA; it is not supported by the current cache. Storing all 576 patch vectors in the same float32 format would increase the current roughly 7.96 GB feature payload to roughly **4.59 TB**. Online extraction is an alternative. A smaller spatial cache is a separate, explicit approximation to assess, not a silent change to the current data contract.

## Minimal experiment to test the recommendation

1. Start a fresh deterministic task readout from epoch 0 on the existing frozen cache. Preserve the historical models. Keep 32D/image, seed 45, identical source windows, targets, adapters, and data schedule.
2. Compare the attention model with a simple nonlinear MLP readout on the same camera-separated inputs. Keep a retuned linear model as the inexpensive reference. The MLP distinguishes improvement from extra nonlinear capacity versus attention itself.
3. Check optimization on source data with a small learning-rate comparison, initially 3e-4 versus 1e-3. Retain the same optimizer method across controls. The old 5e-5 setting is a historical reference; the tiny-batch diagnostic does not establish the best validation LR.
4. Use one representation seed and a short initial run, inspecting epochs 5 and 10 before committing to longer training. Evaluate per-source code/native losses, temporal errors, and mean-code baselines. Verify distinct ordered outputs, missing-camera masking, frozen-backbone gradients, and resume equivalence.
5. Compare MotionTrans using the same frozen-encoder, deep-head, fit/tune episode protocol. Include raw pooled V-JEPA features as a missing reference: this separates a useful frozen backbone from the quality of its learned 32D bottleneck. Report input/head capacity differences. New transfer evaluations require a new launch; the stopped checkpoint sweep stays stopped.
6. If the architecture improves transfer, run the matched stochastic version and assess uncertainty separately. Calibration must use held-out calibration episodes with an explicit predictive distribution/interval procedure; latent sampling is not itself calibration. MotionTrans has already served as a development benchmark, so a stronger final generalization claim eventually needs another untouched target.

## Primary-source basis

The official V-JEPA 2.1 evaluation uses trained attentive probes over frozen features, including transformer blocks and a learned cross-attention query. This supports testing a learned readout, but does not validate the specific width, depth, or tactile design proposed here. Our eight frame queries replace the single classification query to support eight distinct tactile targets.

- [V-JEPA 2.1 paper, especially Appendix C.5](https://arxiv.org/html/2603.14482v1).
- [Pinned official Base evaluation configuration](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/eval_2_1/vitb-384/ssv2.yaml).
