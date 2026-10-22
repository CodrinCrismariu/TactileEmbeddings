# Model implementation

These files contain the actual maintained implementations. Training and model
inference do not import historical implementations or dataset download tools.

Read the files in this order:

1. [`backbone.py`](backbone.py): frozen official V-JEPA checkpoint loading and
   full patch-token extraction. An eight-frame RGB clip produces four temporal
   groups, each containing 24×24 patch vectors per available camera.
2. [`attention.py`](attention.py): the editable shared encoder. It projects each
   768D patch vector to the attention width, adds spatial/time/camera encodings,
   and lets a current-frame query read all available token groups through two
   transformer decoder blocks. An MLP produces the deterministic shared latent.
3. [`joint.py`](joint.py): source-independent encoding followed by a selected
   sensor-code or numeric head. The source identifier never enters attention.
4. [`heads.py`](heads.py): generic adapter and numeric-head construction. Dataset
   selection supplies the dimensions and fixed normalization statistics.
5. [`teachers.py`](teachers.py) and [`autoencoders/`](autoencoders/): frozen native
   reconstruction. Marker teachers use static reference coordinates and surface
   IDs; all other teachers decode their raw sensor code directly.

The experiment configuration selects latent width, attention width/layers/heads,
MLP widths, and sources. Change this implementation when testing a new model
operation; change configuration when selecting dimensions or datasets. Native
tactile sizes and geometry belong to source specifications, not this encoder.

## Baseline compatibility

Existing task checkpoint state dictionaries load strictly, with their original
parameter names and order. `initialization="baseline_v1"` also preserves the
original seed-45 initialization and RNG state. The old constructors consumed
random numbers for a pair projection and two temporary numeric heads before
replacing them; [`initialization.py`](initialization.py) records that detail
explicitly. It affects initialization only and adds no operation to inference.

`autoencoders/factory.py` dispatches recorded checkpoint formats rather than
dataset names. It verifies a supplied teacher hash, loads weights strictly, and
freezes the teacher. Historical format adaptations do not import archive code.

Tests compare seed initialization, dropout/SAM behavior, direct and code heads,
missing camera masks, and teacher formats against archived references. An
additional read-only CPU audit verified the real epoch-3 checkpoint and all 13
accepted source teachers without loading V-JEPA or running GPU training.
