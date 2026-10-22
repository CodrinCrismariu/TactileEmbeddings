"""Render native measurements without resizing or per-episode re-zeroing."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def image_url(array, format="PNG"):
    output = BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(output, format=format)
    mime = "image/jpeg" if format == "JPEG" else "image/png"
    return "data:" + mime + ";base64," + base64.b64encode(output.getvalue()).decode()


def rgb_url(tensor):
    rgb = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return image_url(np.clip(rgb * 255, 0, 255), "JPEG")


def heatmap_url(values, valid, maximum=1.0):
    value = np.clip(np.nan_to_num(values) / maximum, 0, 1)
    stops = np.array(
        [[0, 0, 0], [15, 95, 150], [0, 220, 210], [255, 235, 65], [255, 65, 35]], np.float32
    )
    position = value * (len(stops) - 1)
    lower = np.minimum(position.astype(int), len(stops) - 2)
    mix = (position - lower)[..., None]
    rgb = stops[lower] * (1 - mix) + stops[lower + 1] * mix
    rgb[~valid] = [90, 95, 105]
    return image_url(rgb)


def finite_list(values, valid=None):
    values = np.asarray(values)
    mask = np.isfinite(values)
    if valid is not None:
        mask &= np.asarray(valid, bool)
    return np.where(mask, values, 0.0).tolist()


def target_payload(target, prediction, spec):
    """Return one current target in its native sensor shape and fixed scale."""
    truth = target.values[0, -1].detach().cpu().numpy()
    mask = target.valid[0, -1].detach().cpu().numpy().astype(bool)
    estimate = None if prediction is None else prediction[0, -1].detach().cpu().numpy()
    if estimate is not None and estimate.shape != truth.shape:
        raise ValueError("Decoder output must preserve the measured native target shape")
    expanded = mask
    while expanded.ndim < truth.ndim:
        expanded = np.expand_dims(expanded, -1)
    expanded = np.broadcast_to(expanded, truth.shape) & np.isfinite(truth)
    if estimate is not None and not np.isfinite(estimate).all():
        raise ValueError("Nonfinite decoder prediction")
    mae = (
        None
        if estimate is None or not expanded.any()
        else float(np.abs(estimate - truth)[expanded].mean())
    )
    result = dict(
        kind=target.kind,
        shape=list(truth.shape),
        units=target.units,
        mae=mae,
        valid_count=int(expanded.sum()),
        zero_reference="Fixed source coordinates; no episode-specific re-zeroing",
    )

    if target.kind == "marker_displacement":
        geometry = target.geometry
        count = int(geometry["count"][0])
        reference = geometry["reference"][0, :count].cpu().numpy()
        spacing = geometry["spacing"][0, :count].cpu().numpy()
        measured = truth[:count] * spacing[:, None]
        predicted = None if estimate is None else estimate[:count] * spacing[:, None]
        result.update(
            reference=reference.tolist(),
            surface=geometry["surface"][0, :count].cpu().tolist(),
            values=finite_list(measured),
            valid=mask[:count].tolist(),
            prediction=None if predicted is None else predicted.tolist(),
            units="pixels",
            surface_shape=spec.metadata.get("marker_surface_shape"),
            mae=None
            if predicted is None or not mask[:count].any()
            else float(np.abs(predicted - measured)[mask[:count]].mean()),
        )
        return result

    if target.kind in ("native_numeric", "force_tanh"):
        values = truth.reshape(-1)
        valid = expanded.reshape(-1)
        scale = spec.numeric_transform.get(
            "scale", spec.teacher.get("model_config", {}).get("scale")
        )
        scale = np.ones(len(values)) if scale is None else np.asarray(scale, float)
        if len(scale) != len(values):
            raise ValueError("Numeric display scale does not match native channels")
        groups = spec.metric_groups or {"all": list(range(len(values)))}
        limits = np.ones(len(values))
        for indices in groups.values():
            limits[indices] = max(float(scale[indices].max()) * 3.0, 1e-8)
        if target.kind == "force_tanh":
            limits.fill(1.0)
        result.update(
            values=finite_list(values),
            valid=valid.tolist(),
            prediction=None if estimate is None else estimate.reshape(-1).tolist(),
            labels=spec.metadata.get("channel_order")
            or [f"Channel {i + 1}" for i in range(len(values))],
            limits=limits.tolist(),
            groups=groups,
        )
        return result

    panels = []
    if target.kind == "optical_rgb":
        if truth.ndim != 3 or truth.shape[0] % 3:
            raise ValueError("Optical sensor RGB must retain groups of three native channels")
        for surface in range(truth.shape[0] // 3):
            channels = slice(surface * 3, surface * 3 + 3)
            original = np.clip(truth[channels].transpose(1, 2, 0) * 255, 0, 255)
            reconstructed = (
                None
                if estimate is None
                else np.clip(estimate[channels].transpose(1, 2, 0) * 255, 0, 255)
            )
            panels.append(
                dict(
                    label=f"Surface {surface}",
                    original=image_url(original),
                    prediction=None if reconstructed is None else image_url(reconstructed),
                )
            )
    else:
        channels = truth.reshape(-1, *truth.shape[-2:])
        valid = expanded.reshape(channels.shape)
        reconstructed = None if estimate is None else estimate.reshape(channels.shape)
        layout = spec.metadata.get("taxel_layout")
        for index, values in enumerate(channels):
            hands = spec.metadata.get("hand_order", [])
            channel_label = (
                hands[index].capitalize() + " hand"
                if index < len(hands)
                else f"Channel {index + 1}"
            )
            finger_shape = (
                tuple(layout["shape"][-2:])
                if layout and "shape" in layout
                else ((layout["rows"], layout["cols"]) if layout else None)
            )
            if layout and values.size == len(layout["fingers"]) * np.prod(finger_shape):
                for finger, name in enumerate(layout["fingers"]):
                    shape = finger_shape
                    panel = values.reshape(-1, *shape)[finger]
                    supported = valid[index].reshape(-1, *shape)[finger]
                    predicted = (
                        None
                        if reconstructed is None
                        else reconstructed[index].reshape(-1, *shape)[finger]
                    )
                    panels.append(
                        dict(
                            label=f"{channel_label} · {name}",
                            original=heatmap_url(panel, supported),
                            prediction=None
                            if predicted is None
                            else heatmap_url(predicted, supported),
                        )
                    )
            else:
                panels.append(
                    dict(
                        label=channel_label,
                        original=heatmap_url(values, valid[index]),
                        prediction=None
                        if reconstructed is None
                        else heatmap_url(reconstructed[index], valid[index]),
                    )
                )
    result.update(panels=panels, display_range=[0.0, 1.0], invalid_color="gray")
    return result
