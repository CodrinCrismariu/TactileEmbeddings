# FEEL hand ownership correction

Validation **Lab_P11** records five **right-hand** sensor channels. This applies
to all 19 acquired Lab recordings, across both training and validation splits.

The previous FEEL catalog inherited the Kitchen layout, which contains both hands.
The Lab CSV is named `force_aligned_.csv`, so acquisition preserved its channels as
`released_hand_sensor0`–`released_hand_sensor4` without assigning ownership. The viewer
must not assign the first group to the left hand from its array position.

The [official FEEL project](https://www.cs.umd.edu/~edessale/feel) identifies its Lab
previews as `P<n>_right_hand_*.mp4`, including P11. The pinned
[preprocessing script](https://huggingface.co/datasets/edessa/feel/blob/0f8179649b50afcfd684c87784c67787871db361/preprocess.py)
also applies `parse_ignore(ignore, "right")` to the unnamed Lab files. The local P11
video was visually inspected at six points as a consistency check. Sensor indices
are retained; no unverified sensor-to-finger assignment is made.

`feel_hand_layout.py` now provides explicit ownership per channel. The viewer obtains
groups, channel labels, and trace labels from that episode's mapping. Lab uses one
**Right hand** group and labels `right_sensor0`–`right_sensor4`, with left-hand targets
marked not recorded. An explicitly right-labelled stream remains right even when
it starts at array index 0. Unknown ownership stays unverified instead of defaulting
to left. Only the verified pinned Lab schema receives the unnamed-to-right rule.

RGB, sensor magnitudes, channel ordering, missing masks, frame IDs, and the saved
fixed-zero references are unchanged. Native channel identifiers remain separately
available in the API, preserving the reference/cache contract. No model training or
target rewriting was needed.

Evidence is in `processing/audits/feel_handedness_v1/`: the official page snapshot,
pinned preprocessing script, per-episode ownership metadata, original CSV/NPZ checks,
validation P11 screenshots and browser assertions. `verify.py` checks native samples
and ownership across all 31 acquired FEEL recordings, plus P11's live viewer at frames
0 and 57137.
