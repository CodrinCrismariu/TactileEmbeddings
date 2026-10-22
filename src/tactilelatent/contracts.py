"""Define the sample and batch structures shared by all sensor sources.

The source identifier selects a prediction head after shared visual encoding.
It is not an input to that encoder. All frame mappings are chronological logical
identities; physical camera and sensor indices are retained separately.
Inputs are source-reader tensors and metadata; outputs are typed containers.
Native shapes, units, signed marker displacement and validity masks are retained.
"""

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any


def move_tensors(value, device):
    """Return nested containers with tensors on ``device`` and metadata intact."""
    if hasattr(value, "to") and hasattr(value, "shape"):
        return value.to(device, non_blocking=True)
    if is_dataclass(value):
        return replace(
            value, **{f.name: move_tensors(getattr(value, f.name), device) for f in fields(value)}
        )
    if isinstance(value, dict):
        return {k: move_tensors(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_tensors(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(v, device) for v in value)
    return value


@dataclass
class VisualClip:
    """Scene history and explicit camera availability, without a sensor identity.

    A sample's RGB has shape ``[T, V, 3, H, W]`` with values in ``[0, 1]``;
    a batch adds a leading ``B`` dimension. The maintained reader uses ``T=8``
    and camera slots scene, left wrist, right wrist. ``view_mask`` is ``[T, V]``
    per sample; missing views contain zero RGB and a false mask.
    ``history_frame_indices`` identifies logical source frames, while optional
    ``rgb_frame_indices`` retains the corresponding stored camera frame IDs.
    """

    rgb: Any
    view_mask: Any
    history_frame_indices: Any
    rgb_frame_indices: Any = None
    camera_roles: Any = None
    timestamps: Any = None


@dataclass
class SensorTarget:
    """Native sensor values, support and geometry for the supervised frame(s).

    ``values`` retains its source's dimensions and declared ``units``. ``valid``
    distinguishes missing measurements from measured zero; it may omit trailing
    coordinate dimensions, as with ``[F, N]`` validity for ``[F, N, 2]`` markers.
    Marker geometry includes normalized reference coordinates, surface identity,
    pixel spacing and a point mask. Only variable marker counts are padded.
    """

    values: Any
    valid: Any
    kind: str
    geometry: dict = field(default_factory=dict)
    units: str = "native"

    def expanded_mask(self):
        """Broadcast measurement validity over trailing value coordinates."""
        mask = self.valid
        while mask.ndim < self.values.ndim:
            mask = mask.unsqueeze(-1)
        return mask.expand_as(self.values)


@dataclass
class Supervision:
    """Frozen-teacher codes or direct numeric targets with their own support.

    Code values have shape ``[F, C]`` per sample and are standardized using
    frozen training statistics. ``artifact`` records their cache identity;
    ``orientation`` records whether native targets were mirrored before coding.
    Invalid supervision is excluded by ``valid``, never inferred from zero.
    """

    values: Any = None
    valid: Any = None
    kind: str = "none"
    orientation: str = "original"
    artifact: dict = field(default_factory=dict)


@dataclass
class ClipSample:
    """One source/episode clip with history, current target and provenance.

    Visual history and supervised frame IDs are distinct: current-frame training
    consumes eight images but supervises one final frame. Logical and stored
    sensor frame indices remain separate so release permutations are auditable.
    """

    visual: VisualClip
    target: SensorTarget
    supervision: Supervision
    source: str
    episode_id: str
    split: str
    frame_indices: Any
    sensor_frame_indices: Any
    provenance: dict = field(default_factory=dict)


@dataclass
class ClipBatch:
    """A source- and split-homogeneous batch preserving native target layout.

    Tensor fields add a leading batch dimension to ``ClipSample`` fields.
    ``episode_ids`` and ``provenance`` retain one entry per sample; source identity
    routes the prediction head after shared visual encoding.
    """

    visual: VisualClip
    target: SensorTarget
    supervision: Supervision
    source: str
    episode_ids: list[str]
    split: str
    frame_indices: Any
    sensor_frame_indices: Any
    provenance: list[dict] = field(default_factory=list)

    def to(self, device):
        """Return the same batch structure with tensors moved to ``device``."""
        return move_tensors(self, device)


@dataclass
class SourceSpec:
    """Declare one adapter's target, head, teacher and coordinate conventions.

    ``native_shape`` describes sensor dimensions without a time/batch prefix;
    ``code_dim`` is the teacher's sensor-specific code width. Statistics and
    mirror policies are verified artifact metadata, not learned correspondences.
    Model construction uses this specification without dataset-name branches.
    """

    source: str
    adapter: str
    target_kind: str
    native_shape: tuple
    supervision_kind: str = "code"
    code_dim: int | None = None
    teacher: dict = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)
    numeric_transform: dict = field(default_factory=dict)
    mirror: dict = field(default_factory=dict)
    metric_groups: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def head_dimension(self):
        """Return the cached-code width or flattened direct-target width."""
        if self.supervision_kind == "code":
            return self.code_dim
        result = 1
        for n in self.native_shape:
            result *= n
        return result
