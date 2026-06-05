#!/usr/bin/env python3
"""Write a development-only image-grounded constraint-program demo artifact.

This does not call a VLM or simulator. It validates the new ToolFrame data path:
2D keypoints + depth/camera -> lifted 3D refs -> staged constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parttoolgen.image_grounded import (
    CameraIntrinsics,
    ImageKeypoint,
    build_contact_motion_program,
    derive_axis,
    lift_keypoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="artifacts/image_grounded/demo_contact_motion",
        help="Output directory for lifted refs and constraint program.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = CameraIntrinsics(width=640, height=480, fx=580.0, fy=580.0, cx=320.0, cy=240.0)
    camera_to_world = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    start = lift_keypoint(
        ImageKeypoint("start", 310.0, 240.0, "scene_start", 0.9),
        depth_m=0.75,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )
    end = lift_keypoint(
        ImageKeypoint("end", 390.0, 240.0, "scene_end", 0.9),
        depth_m=0.75,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )
    direction = derive_axis(start.position_world, end.position_world)
    program = build_contact_motion_program(
        task_id="demo_contact_motion",
        instruction="Use the tool's selected contact point to move from start to end.",
        tool_refs={
            "contact_point_face_id": "F_demo",
            "contact_point_ref": "tool.contact_point",
            "motion_axis_ref": "tool.motion_axis",
        },
        scene_refs={
            "start": start.to_json(),
            "end": end.to_json(),
            "direction": direction,
        },
    )

    (out_dir / "lifted_refs.json").write_text(
        json.dumps(
            {
                "schema_version": "toolframe_lifted_refs_v0",
                "camera_intrinsics": intrinsics.__dict__,
                "start": start.to_json(),
                "end": end.to_json(),
                "direction": direction,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "constraint_program.json").write_text(
        json.dumps(program.to_json(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
