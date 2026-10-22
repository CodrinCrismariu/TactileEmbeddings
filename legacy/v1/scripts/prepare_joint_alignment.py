#!/usr/bin/env python3
"""Prepare, pin, and audit the new joint experiment without starting training."""
import argparse
import json

from tactilelatent.aligned_frames import MANIFEST, prepare_manifest
from tactilelatent.audit_alignment import audit_manifest
from tactilelatent.sensor_code_cache import REGISTRY, pin_teachers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="processing/joint_all/manifest_sensor_roles_corrected_v2.json")
    parser.add_argument("--frame-order", default="processing/audits/vla_touch_native_dynamics/dataset_frame_order.json")
    parser.add_argument("--camera-roles", default="configs/freetacman_camera_roles.json")
    parser.add_argument("--output", default=MANIFEST)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--audit", default="processing/joint_alignment/audit.json")
    args = parser.parse_args()
    prepare_manifest(args.original, args.frame_order, args.camera_roles, args.output)
    pin_teachers(args.registry)
    print(json.dumps(audit_manifest(args.output, args.audit)), flush=True)


if __name__ == "__main__":
    main()
