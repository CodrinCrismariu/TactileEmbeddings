"""Official OpenTouch row/column-to-hand mapping, for display only.

Coordinates retain the upstream orientation (x right, y down). Model inputs
remain raw / 3072 on the native 16x16 electrical channel grid.
"""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def hand_layout():
    asset = json.loads(Path(__file__).with_name("opentouch_hand_layout.json").read_text())
    expected = {f"{r}-{c}" for r in range(16) for c in range(16)}
    excluded = set(asset["erasedNodes"])
    if set(asset["positions"]) != expected or len(excluded) != 87 or not excluded <= expected:
        raise ValueError("Invalid official OpenTouch channel mapping")
    nodes = []
    for key, position in asset["positions"].items():
        if key in excluded:
            continue
        row, column = map(int, key.split("-"))
        x, y = float(position["x"]), float(position["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Invalid OpenTouch hand coordinate")
        nodes.append({"id": key, "row": row, "column": column,
                      "index": row * 16 + column, "x": x, "y": y})
    nodes.sort(key=lambda node: node["index"])
    return {"id": asset["id"], "hand": "right", "nodes": nodes,
            "mapped_channels": len(nodes), "excluded_channels": len(excluded),
            "source_url": asset["source_url"], "source_sha256": asset["source_sha256"],
            "bounds": [min(n["x"] for n in nodes), min(n["y"] for n in nodes),
                       max(n["x"] for n in nodes), max(n["y"] for n in nodes)]}


def mapped_mask():
    mask = np.zeros((1, 16, 16), dtype=bool)
    for node in hand_layout()["nodes"]:
        mask[0, node["row"], node["column"]] = True
    return mask
