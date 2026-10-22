# Baihu autoencoder failure and repair

Date: 2026-10-19. This investigation concerns the **sensor autoencoder's 32D
code**, not the shared visual model's latent. No joint visual training was run.

## Finding

The failed model had stopped encoding its input. Across native moving clips
from all three platforms, every frame produced the exact same code. Replacing
all measured vectors with zeros left its predictions unchanged.

The immediate failure was the hidden GELU activation before the sensor code:
its pre-activations were approximately **−70.82 to −6.93**. All 128 hidden
outputs evaluated to zero at the stored checkpoint's numerical precision.
Consequently the 32D code became the final linear layer's bias. Gradient norms
through most of the encoder were approximately `1e-17`–`1e-15`; the last code
weight matrix had exactly zero gradient. The decoder could still update its
static geometry/surface response, which explains small changes in validation
loss without learning interaction-dependent reconstruction.

The training sampler was also problematic. It shuffled **episodes**, then fed
chronological batches from one episode at a time. At the 100,000-frame budget,
epoch 1 visited only 231 of the 2,710 training episodes. Across all 30 epochs,
194 episodes were never visited. The working older marker AE trainer instead
used a globally sampled frame pool and shuffled frames across episodes.

The results below support batching as an important optimization problem.
Normalization addresses the observed saturation failure. This investigation
does not establish that every original run would collapse under every seed.

## Controlled diagnosis

Four short arms used identical native samples, identical initial weights for
shared components, seed 45, AdamW at `1e-3`, and the unchanged native marker
loss. They used 8,448 frames from 66 training episodes and 2,816 frames from 22
disjoint validation episodes, spanning the task/platform groups. Each ran eight
pilot epochs. Batch sizes and numbers of updates were matched across batching
conditions. No target images, marker coordinates, or vectors were resized.

| Code projection | Batches | Validation MAE, px |
|---|---|---:|
| Original | One episode at a time | 0.252809 |
| Original | Mixed episodes | 0.210774 |
| LayerNorm before GELU | One episode at a time | 0.253170 |
| LayerNorm before GELU | Mixed episodes | 0.215483 |
| Zero displacement | — | 0.252138 |

Mixing episodes restored useful learning in this pilot. Normalization alone
did not improve held-out reconstruction. The original mixed model was slightly
better than the normalized mixed model at this short budget; normalization was
retained to guard against the complete hidden-activation failure observed in
the production checkpoint, not because it won this pilot's validation score.

On a separate 32-frame moving native clip, 400 fitting steps reduced MAE from
the 1.259843-px zero baseline to 0.891679 px with the original model and
0.697158 px with normalization. A target-conditioned affine-field oracle using
24 coefficients achieved 0.130746 px on the pilot validation frames, versus
0.252138 px for zero. This oracle is a capacity diagnostic, not a learned
teacher: meaningful compressible variation exists in the stored vectors.

The matched pilot chart is
`processing/audits/baihu_autoencoder_investigation_v1/matched_pilot.png`.

## Implemented repair

Only the sensor-code projection and training sampling changed:

```text
Original: attention → flatten → Linear(1024,128) → GELU → Linear(128,32)
Repair:   attention → flatten → Linear(1024,128) → LayerNorm(128)
                                                              → GELU → Linear(128,32)
```

The native four-surface marker encoder/decoder, 32D sensor code, width 128,
eight attention queries, four heads, signed outputs, and original reconstruction
objective are preserved. LayerNorm acts before GELU so a large negative offset
cannot turn the entire hidden vector off.

The repair samples a fixed 100,000-frame training pool uniformly without
replacement, then shuffles frames across episodes every epoch. This particular
pool covers all 2,710 training episodes. Padding affects only the variable marker
axis and remains masked. Target displacement retains native precision and its
existing marker-spacing units; reported errors convert back to native pixels.

The run uses seed 45, 30 epochs, AdamW `1e-3`, weight decay `1e-4`, and cosine
decay to `1e-6`. Its **9,999 validation frames from 303 episodes exactly match
the failed run's frame IDs**, with 6,477,072 valid displacement coordinates.
There is no train/validation episode overlap. No hyperparameter sweep or
additional seeds were launched.

New code lives in `baihu_autoencoder.py` and `baihu_ae_data.py`; the loader has a
distinct `expansion_marker_autoencoder_v2` checkpoint type. Historical two- and
four-surface models retain their existing classes and remain loadable.

## Full validation results

The repaired run completed all 30 epochs; its best checkpoint is epoch 30.

| Metric, native pixels | Failed v2 | Repaired v3 |
|---|---:|---:|
| Overall marker MAE | 0.266962 | **0.087232** |
| Active-marker MAE | 2.183921 | **0.481916** |
| Sampled trajectory-change MAE | 0.130485 | **0.064015** |
| D-WHEEL MAE | 0.203275 | **0.075264** |
| PIKA MAE | 0.124591 | **0.064868** |
| QINGLOONG MAE | 1.029301 | **0.219357** |

Overall MAE improves by approximately **67.3%**. All four individual surfaces
also improve. Active markers use the existing threshold of half the train-only
displacement scale; their zero baseline is 2.187802 px. Trajectory change here
means changes between the sampled validation frames, not a full 30-fps dynamics
evaluation.

The repaired codes have within-episode standard deviation approximately 0.413
and effective rank 9.46; the failed codes had zero spread and rank zero. These
are diagnostics of the tactile teacher, not evidence of sensor invariance or
transfer improvement.

## Does the code carry changing interaction information?

The frozen teacher was tested with unchanged native geometry while its codes
were shuffled **within each validation episode**:

| Decoder input | Failed v2 MAE, px | Repaired v3 MAE, px |
|---|---:|---:|
| Correct frame's code | 0.266962 | **0.087232** |
| Another frame's code from the same episode | 0.266962 | 0.366411 |
| Episode-mean code | 0.266962 | 0.277165 |

The failed decoder is unaffected because its codes are constant. The repaired
decoder needs the correct frame's code to reconstruct changing displacement.
Replacing measured input vectors with zero changes its output by 0.230776 px
on average, compared with exactly zero change for the failed teacher.

This establishes reconstruction of the stored displacement trajectories. It
does not independently revalidate marker extraction against the retired tactile
videos or establish a physical mirror mapping.

## Artifacts and verification

- Failed teacher preserved: `processing/data_expansion_v1/teachers/baihu_native_v2/`.
- Repaired teacher: `processing/data_expansion_v1/teachers/baihu_native_v3/`.
- Viewer entry: **baihu AE · 32D · native v3 · normalized code projection**.
- Chronological frozen codes: `processing/data_expansion_v1/codes/baihu_native_v3/`.
- Investigation, activation traces, exact frame selections, per-platform and
  per-surface results, and frame-code ablations:
  `processing/audits/baihu_autoencoder_investigation_v1/`.
- Training/caching service: `tactilelatent-baihu-ae-repair-v3.service`.

The quality gate requires at least 10% improvement over zero on overall and
active MAE, measurable input dependence, and within-episode code variation.
The repaired teacher passed all four checks. Its code cache is complete for
**all 3,013 episodes / 2,564,143 frames**, with fresh producer-batch replay
matching stored codes exactly at the checked train/validation boundaries.
The accepted teacher and verified cache are active in the prepared 12-source
registry. Training plus code preparation took approximately 4.6 minutes.
Historical teachers
and code caches are preserved; EgoPressure remains task-only and UV-only;
MotionTrans remains held out; physical mirroring eligibility is unchanged.

The focused regression suite passed **55 tests**. Checks cover the saturation
mechanism, gradient flow through the normalized code head, checkpoint loading
for new and historical architectures, native temporal sampling, padding/missing
observations, and producer-batch code-cache replay. The repaired frozen teacher
also passed a real 325-marker interface check and a browser reconstruction check.

Reproduction:

```bash
.venv/bin/python scripts/investigate_baihu_autoencoder.py
.venv/bin/python scripts/audit_baihu_autoencoder_repair.py
.venv/bin/python -m pytest tests/test_baihu_autoencoder.py tests/test_data_expansion.py
```
