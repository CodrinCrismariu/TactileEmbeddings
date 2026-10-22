# EgoPressure pressure visualization

Select **EgoPressure** on the sensor reconstruction page. The **3D hand surface**
opens by default with the installed UV meshes. **Enlarge comparison** opens the
measured pressure, reconstruction and absolute error together, with a frame
slider and the corresponding scene thumbnail. **Pressure display → UV atlas**
returns to the flat maps, where hovering places synchronized crosshairs across
all three maps. The optional grid marks every 32 native texels. Escape closes
the enlarged view.

**Color contrast** changes only the display curve. A value below 1 makes weak
pressure easier to see; 1 restores a linear color scale. All three comparisons
use the same curve and the existing pressure/error scale controls. Targets,
metrics, training, native 224×224 resolution, and hover values are unchanged.

## Exact 3D hand view

Pressure is mapped onto the recorded hand mesh using the actual MANO UV topology,
joining UV chart cuts on the hand surface. Both hand files are installed and the
real rendering has been verified. No approximate hand or inferred UV-to-finger
assignment is substituted.

The installed **MANO_UV_left.obj** and **MANO_UV_right.obj** came from the public
[EgoForce author release](https://huggingface.co/chris10/EgoForce/tree/5a57b4026babc5b35cb53627c2e1c709e1276b65/_DATA/mano),
which identifies these assets as the official MANO distribution. The pinned
revision is `5a57b4026babc5b35cb53627c2e1c709e1276b65`; checksums and the upstream
license notice are saved under `processing/audits/egopressure_surface_view_v2/`.
The two files total 201,087 bytes and retain the MANO non-commercial research
license. Their installed locations are:

```text
TactileLatent/mano_models/MANO_UV_left.obj
TactileLatent/mano_models/MANO_UV_right.obj
```

Alternatively set `EGOPRESSURE_MANO_PATH` in the viewer service environment to
their directory. The local directory is checked on **Check hand models**;
installing files there does not require a viewer restart.

This viewer only needs the two UV OBJ files. Posed hand vertices and joints
already come from each dataset frame, so MANO pickle files and a MANO forward
model are unnecessary for this display.

Refresh an already open page to load the new default. Drag any 3D panel to
rotate all three together, scroll to zoom, or use **Reset view** and **Turn hand
over**. The default orientation shows the contact side. All panels use the same
recorded hand geometry. Only their pressure textures differ; the autoencoder is
not being credited with predicting pose. An explicit UV-atlas selection persists
while browsing that page.

The UV coordinates follow the official convention:

```text
native column = u × 223
native row    = (1 − v) × 223
```

Vertex and texture-face indices stay separate at UV seams. Geometry is joined
using the selected frame's native `source_frame_id` and checked against its hand
side. Frames without a recorded pose show the UV atlas instead of reusing a
neighboring pose. Missing pressure remains gray. Rendering runs locally in the
browser and needs no model retraining or extra dataset cache.

## Verification

The UV enlargement, contrast invariance of MAE, hover, dataset switching and
Escape handling passed live browser checks. The shader, linked rotation, and
missing-pose fallback passed browser tests with explicitly synthetic geometry.
There were no browser script errors. **43 focused tests passed**, covering UV
seam indices, bottom-origin UV versus top-origin pixels, missing/invalid assets,
native annotation-frame joins, hand-side checks and existing viewer behavior.

Live browser checks now use the actual left and right UV meshes and recorded
EgoPressure poses: 778 vertices, 1,538 triangles and 891 texture coordinates per
hand. All measured/reconstructed/error panels render with zero WebGL errors.
Native frame joins, the atlas switch, and contact visibility on both hands were
checked. Evidence and screenshots are under
`processing/audits/egopressure_surface_view_v2/`. The earlier synthetic renderer
checks remain under `egopressure_surface_view_v1/`.
