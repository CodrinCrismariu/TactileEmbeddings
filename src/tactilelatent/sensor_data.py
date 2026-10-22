"""Native sensor targets and scene frames, without requiring RGB for pretraining.

Normalization matches JointClipDataset. Optical targets remain RGB; signed force
and torque retain the existing tanh representation (raw zero maps to 0.5).
"""
from io import BytesIO

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .joint_data import _ftp_group, _from_uint8_images, _json_records
from .media import video_frames, video_info


SENSORS = {
    "egotouch": ("EgoTouch", "Bilateral hand pressure", ["Left hand", "Right hand"]),
    "egotactile": ("EgoTactile", "Right-hand pressure glove · 256 stored taxels", ["Right-hand taxel vector"]),
    "opentouch": ("OpenTouch", "Right-hand pressure array", ["Right hand"]),
    "opengraph": ("OpenGraph", "Bilateral OGLO glove taxel vectors", ["Left hand", "Right hand"]),
    "freetacman": ("FreeTacMan", "Two optical tactile surfaces", ["Surface 1", "Surface 2"]),
    "ftp_unit_bimanual": ("FTP Unit Bimanual", "Bilateral optical gripper tactile", ["Left gripper", "Right gripper"]),
    "ftp_unit": ("FTP Unit", "Right optical gripper tactile", ["Right gripper"]),
    "ftp_vla_touch": ("FTP VLA Touch", "Right optical gripper tactile", ["Right gripper"]),
    "ftp_vitamin": ("FTP ViTaMIn", "Two right-gripper optical images", ["Right gripper · sensor 1", "Right gripper · sensor 2"]),
    "ftp_rdp": ("FTP RDP", "Signed gripper-force channel", ["Left gripper force"]),
    "ftp_rdp_bimanual": ("FTP RDP Bimanual", "Signed bilateral gripper force", ["Left force", "Right force"]),
    "ftp_motiontrans": ("FTP MotionTrans", "Six finger/thumb torque channels", ["Finger 1", "Finger 2", "Finger 3", "Finger 4", "Thumb 1", "Thumb 2"]),
}


def ftp_group(row):
    return _ftp_group(row["tar_path"], row["tar_index"],
                      row.get("archive_group", "Unit_Bimanual"), row["zarr_name"])


def load_targets(row, indices):
    """Return normalized targets [N,C,H,W], retaining NaN invalid cells."""
    indices = np.asarray(indices, dtype=np.int64)
    source = row["source"]
    if row["target_kind"] == "optical_rgb":
        from .native_optical import native_targets
        return native_targets(row, indices)
    if source == "egotouch":
        with np.load(row["target_path"]) as data:
            target = torch.from_numpy(np.stack([data[f"{side}_pressure_grid"][indices]
                                                 for side in ("left", "right")], axis=1)).float()
    elif source == "egotactile":
        records = _json_records(row["target_path"])
        target = torch.tensor([records[int(i)]["RH"]["sensor_256"] for i in indices],
                              dtype=torch.float32).reshape(len(indices), 1, 1, 256)
    elif source == "opentouch":
        with h5py.File(row["views"]["0"], "r") as h5:
            target = torch.from_numpy(h5["data"][row["hdf5_demo"]]["right_pressure"][indices].copy()).float().unsqueeze(1)
    elif source == "opengraph":
        target = torch.from_numpy(np.asarray(np.load(row["target_path"], mmap_mode="r")[indices + row["target_start"]]).copy()).float().unsqueeze(2)
    elif "tar_path" in row:
        group = ftp_group(row)
        positions = indices + row["target_start"]
        keys = row.get("target_keys", ["left_tactile_data_gripper", "right_tactile_data_gripper"])
        raw = torch.cat([torch.from_numpy(np.asarray(group[f"data/{key}"][positions]).copy())
                         .float().reshape(len(indices), -1) for key in keys], dim=1)
        target = (.5 + .5 * torch.tanh(raw / row["target_transform_scale"])).reshape(len(indices), -1, 1, 1)
    else:
        raise ValueError(f"Unsupported target source: {source}")
    if row["target_kind"] != "optical_rgb":
        target = target / row["target_scale"]
    valid = torch.isfinite(target)
    if list(target.shape[1:]) != row["target_shape"] or not valid.any():
        raise ValueError(f"Invalid target shape or empty sensor values: {source}:{row['episode_id']}")
    if ((target[valid] < 0) | (target[valid] > 1)).any():
        raise ValueError(f"Sensor target outside [0,1]: {source}:{row['episode_id']}")
    # Keep invalid values recognizable in the disk cache.
    return torch.where(valid, target, torch.nan).numpy()


def scene_frames(row, indices, camera, size=480):
    indices = np.asarray(indices, dtype=np.int64)
    if camera not in row["views"]:
        raise ValueError("Camera is unavailable for this episode")
    from .storage import converted_scene
    converted = converted_scene(row, indices, size)
    if converted is not None:
        return converted[0][:, int(camera)]
    if "tar_path" in row:
        images = _from_uint8_images(ftp_group(row)[f"data/{row['views'][camera]}"][indices + row["target_start"]])
        return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    if row["source"] == "opentouch":
        images = []
        with h5py.File(row["views"][camera], "r") as h5:
            demo = h5["data"][row["hdf5_demo"]]
            for i in indices:
                with Image.open(BytesIO(bytes(demo["rgb_images_jpeg"][int(i)]))) as image:
                    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
                    images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.)
        return torch.stack(images)
    if row["source"] == "opengraph":
        indices = indices + row["video_start"]
    return video_frames(row["views"][camera], indices, size=size)


def scene_metadata(row, camera):
    if camera not in row["views"]:
        raise ValueError("Camera is unavailable for this episode")
    from .storage import scene_entry
    entry = scene_entry(row)
    if entry is not None:
        return {'video_frames': row['frames'], 'fps': entry['views'][str(camera)]['fps']}
    if "tar_path" in row or row["source"] == "opentouch":
        # A sample index is not a verified source-video timestamp.
        return {"video_frames": row["frames"], "fps": None}
    frames, fps = video_info(row["views"][camera])
    return {"video_frames": frames, "fps": fps}
