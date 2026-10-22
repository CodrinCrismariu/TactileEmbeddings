> Historical record, relocated on October 21, 2026. Original path: `VLA_TOUCH_DIRECT_TRAINING.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Direct VLA_touch image-to-tactile experiment

**Historical experiment, retained unchanged.** The current direct control
trains on MotionTrans; VLA_touch is now part of joint baseline training.
See [MOTIONTRANS_BASELINE_TRAINING.md](MOTIONTRANS_BASELINE_TRAINING.md).

This run answers whether the project's image encoder and 32D Gaussian latent
can learn VLA_touch when its tactile labels are available during encoder
training. It keeps DINOv3 frozen, but trains the camera-fusion, temporal and
spatial transformers, Gaussian bottleneck, and 32→512→3072 optical decoder
together from fresh weights. It does not initialize from either prior model.

The manifest has 1,523 train and 183 validation non-overlapping 8-frame clips.
Each of 50 epochs uses all 1,523 train clips (the 8,000-clip cap is therefore
inactive). Validation covers all 183 clips, rather than a 64-clip subsample.
The decoder and latent objective use the same normalized optical-image loss as
the joint trainer, with KL weight ramping to 0.001 over ten epochs. The final
checkpoint is evaluated, not a validation-selected checkpoint.

After training, `scripts/train_vla_touch_direct.sh` runs the same deterministic
and 64-draw pixel/clip evaluation routine used by both frozen-encoder probes.
It then writes `evaluation/three_way_comparison.json`, asserting that all three
evaluations share the manifest, 183 validation clips, and 64 draws. The direct
run is **not a zero-shot transfer evaluation**: its encoder and decoder both
saw VLA_touch train labels. The two prior encoders did not see VLA_touch during
their joint training; only their new probe decoders saw its train labels.

```bash
bash scripts/train_vla_touch_direct.sh
tail -f processing/runs/vla_touch_direct_vae32_50ep/nohup.log
```

For an interrupted run, use `bash scripts/train_vla_touch_direct.sh --resume`.
The detached run's TensorBoard is on remote port 6009; forward it in VS Code's
Ports panel before opening it in a local browser. Completed evaluation results
refuse overwrite by default.
