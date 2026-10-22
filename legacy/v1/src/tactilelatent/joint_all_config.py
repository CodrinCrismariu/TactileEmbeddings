"""Native target definitions for the optional twelve-source VAE experiment."""

from .prepare_joint import TARGETS as ORIGINAL_TARGETS


# Each decoder predicts one explicit target layout. RDP targets measured
# gripper force; MotionTrans predicts the six recorded torque channels.
# Optical targets retain the sensor images, resized to 32x32 by the loader.
EXTRA_TARGETS = {
    "ftp_vla_touch": ("optical_rgb", [3, 32, 32], 255.0),
    "ftp_unit": ("optical_rgb", [3, 32, 32], 255.0),
    "ftp_rdp_bimanual": ("force_tanh", [2, 1, 1], 1.0),
    "ftp_rdp": ("force_tanh", [1, 1, 1], 1.0),
    "ftp_motiontrans": ("torque_tanh", [6, 1, 1], 1.0),
    "ftp_vitamin": ("optical_rgb", [6, 32, 32], 255.0),
}
ALL_TARGETS = {**ORIGINAL_TARGETS, **EXTRA_TARGETS}


FTP_SPECS = {
    "ftp_vla_touch": {
        "group": "VLA_touch", "views": {"0": "camera_main_rgb", "2": "right_wrist_camera_rgb"},
        "target_keys": ["right_tactile_data_gripper"],
    },
    "ftp_unit": {
        "group": "Unit", "views": {"0": "camera_main_rgb", "2": "right_wrist_camera_rgb"},
        "target_keys": ["right_tactile_data_gripper"],
    },
    "ftp_rdp_bimanual": {
        "group": "RDP_Bimanual",
        "views": {"0": "camera_ego_rgb", "1": "left_wrist_camera_rgb",
                  "2": "right_wrist_camera_rgb"},
        "target_keys": ["left_tactile_data_gripperforce_flexivgripper",
                        "right_tactile_data_gripperforce_flexivgripper"],
        "target_transform_scale": 30.0,
    },
    "ftp_rdp": {
        "group": "RDP", "views": {"0": "camera_ego_rgb", "1": "left_wrist_camera_rgb"},
        "target_keys": ["left_tactile_data_gripperforce_flexivgripper"],
        "target_transform_scale": 30.0,
    },
    "ftp_motiontrans": {
        "group": "MotionTrans", "views": {"0": "camera_ego_rgb"},
        "target_keys": ["right_tactile_data_fingertorque",
                        "right_tactile_data_thumbtorque"],
        "target_transform_scale": 1000.0,
    },
    "ftp_vitamin": {
        "group": "ViTaMIn", "views": {"2": "right_wrist_camera_rgb"},
        "target_keys": ["right_tactile_data_gripper"],
    },
}
