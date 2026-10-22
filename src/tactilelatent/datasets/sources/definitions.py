"""Declared sensor semantics, channel groups and release adapters.

Adding a prepared source registers a definition and its reader. The registry,
collator, sampler, model and training loop do not acquire a new source branch.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceDefinition:
    """Describe a release's reader, native representation and metric groups.

    ``adapter_factory`` optionally supplies the common adapter interface for a
    prepared source. Shape, channel groups and acquisition filters belong here,
    keeping model and training code independent of dataset names.
    """

    source: str
    target_kind: str
    adapter: str = "prepared_manifest"
    native_shape: tuple | None = None
    acquisition: str | None = None
    row_filters: dict = field(default_factory=dict)
    input_dim: int | None = None
    metric_groups: dict = field(default_factory=dict)
    native_metric_reduction: str = "batch"
    code_family: str = "contiguous"
    adapter_factory: object = None
    metadata: dict = field(default_factory=dict)

    def metadata_for_row(self, row):
        """Return source metadata augmented with this row's native geometry."""
        result = dict(self.metadata)
        if self.target_kind == "marker_displacement":
            result["marker_surface_shape"] = list(
                row.get("native_sensor_shape", row["target_shape"][-2:])
            )
        return result


_DEFINITIONS = {}


def register_source(definition, replace=False):
    """Register sensor semantics, rejecting duplicate names unless requested."""
    if definition.source in _DEFINITIONS and not replace:
        raise ValueError(f"Source already registered: {definition.source}")
    if definition.native_metric_reduction not in ("cells", "batch"):
        raise ValueError("Unknown native metric reduction")
    _DEFINITIONS[definition.source] = definition
    return definition


def source_definition(source):
    """Return a registered release definition or reject an unknown source."""
    try:
        return _DEFINITIONS[source]
    except KeyError:
        raise ValueError(f"No source definition registered: {source}") from None


for name, kind in (
    ("egotouch", "pressure_grid"),
    ("egotactile", "taxels"),
    ("opentouch", "taxels"),
    ("opengraph", "taxels"),
    ("ftp_vitamin", "optical_rgb"),
    ("ftp_rdp", "force_tanh"),
    ("ftp_rdp_bimanual", "force_tanh"),
    ("ftp_motiontrans", "force_tanh"),
):
    register_source(
        SourceDefinition(
            name,
            kind,
            metadata=(
                {
                    "taxel_layout": {
                        "fingers": ["index", "middle", "pinky", "ring", "thumb"],
                        "shape": [5, 4, 4],
                    },
                    "hand_order": ["left", "right"],
                }
                if name == "opengraph"
                else {"hand_order": ["left", "right"]}
                if name == "egotouch"
                else {}
            ),
        )
    )
for name in ("ftp_unit", "ftp_unit_bimanual", "ftp_vla_touch", "baihu"):
    register_source(
        SourceDefinition(
            name,
            "marker_displacement",
            native_shape=(-1, 2),
            code_family="expansion" if name == "baihu" else "contiguous",
        )
    )
register_source(
    SourceDefinition(
        "egopressure",
        "pressure_grid",
        native_shape=(1, 224, 224),
        native_metric_reduction="cells",
        code_family="expansion",
    )
)
for name, release, filters, dimension, groups in (
    ("feel_lab", "feel", {"environment": "Lab"}, 5, {}),
    ("feel_kitchen", "feel", {"environment": "Kitchen"}, 12, {}),
    ("hrdexdb_allegro_v5", "hrdexdb", {"embodiment": "allegro_v5"}, 4, {}),
    (
        "hrdexdb_inspire_f1",
        "hrdexdb",
        {"embodiment": "inspire_f1"},
        24,
        {
            name: list(range(i, 24, 3))
            for i, name in enumerate(("normal", "tangential", "direction"))
        },
    ),
    (
        "trex",
        "trex",
        {},
        60,
        {
            "force": [i for i in range(60) if i % 6 < 3],
            "moment": [i for i in range(60) if i % 6 >= 3],
        },
    ),
):
    register_source(
        SourceDefinition(
            name,
            "native_numeric",
            "native_acquisition",
            (dimension,),
            release,
            filters,
            dimension,
            groups,
            "cells",
            "acquisition",
        )
    )

register_source_definition = register_source
