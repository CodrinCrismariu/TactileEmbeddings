> Historical record, relocated on October 21, 2026. Original path: `FTP_EXTRA_DATASETS.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Additional FTP-1 data

The completed FTP-1 expansion uses pinned Hugging Face revision
`d287c9c532514a08b2eb29bf1259997bb32f4f3b`. In addition to the
previously verified `Unit_Bimanual` archive, the selected archives are
`VLA_touch`, `Unit`, `RDP_Bimanual`, `RDP`, `MotionTrans`, and `ViTaMIn`.
Their combined archive size is 191,132,641,280 bytes, and each archive
matched its pinned upstream LFS SHA-256. The six additions contribute
**3,255 paired episodes / 835,556 frames**. With the existing data, the
measured `datasets` + `processing` total is about **649 GB decimal**, below
the user's 700 GB cap.

| Archive group | Size (GB decimal) | Paired episodes | Paired frames | Visual/tactile targets |
| --- | ---: | ---: | ---: | --- |
| VLA_touch | 9.90 | 380 | 28,272 | Main/wrist RGB + optical tactile RGB |
| Unit | 13.71 | 170 | 38,585 | Main/wrist RGB + optical tactile RGB |
| RDP_Bimanual | 14.25 | 50 | 23,251 | Ego/two wrists + bilateral GelSight, MCTac, and gripper force |
| RDP | 33.74 | 140 | 62,977 | Ego/left wrist + GelSight, MCTac, and gripper force |
| MotionTrans | 25.57 | 1,768 | 321,132 | Ego RGB + finger/thumb torque readings |
| ViTaMIn | 93.96 | 747 | 361,339 | Wrist RGB + two optical tactile RGB images |

Only the archive groups relevant to camera/tactile data are downloaded.
Archives are retained in their compressed tar form and indexed for random
Zarr access, so no second extracted copy is stored. **The archives themselves
still contain ancillary robot state/actions; they have not been repacked to
remove those fields.** We do not read those fields for image-to-tactile work.

Download integrity logs/reports:

- `processing/audits/ftp_extra_download.log`
- `processing/audits/ftp_extra_download_second.log`
- `processing/audits/ftp_extra_download.json`
- `processing/audits/ftp_extra_download_second.json`

Array indexes are `processing/indexes/ftp_<group>.json`. Detailed paired
camera/tactile array schema and three sampled frame reads per array are in
`processing/audits/ftp_extra_schema.json`. All 45 Zarr task groups had
aligned camera/tactile first dimensions and passed sampled finite-value
checks. The index and schema audit commands are:

```bash
.venv/bin/python scripts/index_ftp_archives.py VLA_touch Unit RDP_Bimanual RDP MotionTrans ViTaMIn
.venv/bin/python scripts/audit_ftp_extra_schema.py VLA_touch Unit RDP_Bimanual RDP MotionTrans ViTaMIn --sample-frames
```

The original six-source configuration remains unchanged. A separate
12-source manifest and 11-source training configuration now consume these
archives; see `JOINT_ALL_TRAINING.md`. In that historical run, `VLA_touch` was held out entirely from
that training run for later frozen-encoder adaptation. The native tactile
layouts and units differ, so each training source has an explicit target,
normalization, and decoder. In particular, `RDP_Bimanual` has both optical
tactile cameras and numerical gripper-force readings; this first run targets
the force values and does not collapse them into a generic pressure image.

The current design instead holds out **MotionTrans** and includes VLA_touch
in joint training. See [MOTIONTRANS_BASELINE_TRAINING.md](MOTIONTRANS_BASELINE_TRAINING.md)
for fresh baseline training, the sequential MotionTrans-only control and
six-channel torque evaluation. Existing archive formats and targets stay unchanged.
