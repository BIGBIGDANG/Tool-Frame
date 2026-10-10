"""Software regressions for the cleaned method; no task scoring or model calls."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ToGo.image_grounded import CameraIntrinsics, build_contact_motion_program
from ToGo.image_grounded.staged_mainline_adapter import build_mainline_inputs_from_image_grounded_program
from ToGo.pipeline.handoff import ROOT, compact_tool_refs, read_json, write_json
from ToGo.pipeline.lifting import build_program, lift_to_plane
from ToGo.execution.simtoolreal_rollout import environment_overrides


def test_oblique_ray_plane_lift_preserves_pixel_and_plane():
    intrinsics = CameraIntrinsics(100, 100, 80, 80, 50, 50)
    camera = np.eye(4)
    plane = {"point_world": [0, 0, 2], "normal_world": [0.4, 0, -1]}
    point = np.asarray(lift_to_plane(u=70, v=60, intrinsics=intrinsics,
                                     camera_to_world=camera, plane=plane))
    assert np.dot(point - plane["point_world"], plane["normal_world"]) == pytest.approx(0, abs=1e-10)
    assert 80 * point[0] / point[2] + 50 == pytest.approx(70)
    assert 80 * point[1] / point[2] + 50 == pytest.approx(60)


@pytest.mark.parametrize("normal,point,message", [([1, 0, 0], [1, 0, 0], "parallel"),
                                                 ([0, 0, 1], [0, 0, -1], "behind")])
def test_invalid_ray_plane_geometry_is_rejected(normal, point, message):
    with pytest.raises(ValueError, match=message):
        lift_to_plane(u=50, v=50, intrinsics=CameraIntrinsics(100, 100, 80, 80, 50, 50),
                      camera_to_world=np.eye(4), plane={"point_world": point, "normal_world": normal})


def test_bent_path_keeps_endpoint_and_explicit_normal_axis():
    program = build_contact_motion_program(task_id="bend", instruction="Follow the marked bend",
        tool_refs={"tool_id": "unit", "contact_point_tool": [0, 0, 0], "motion_axis_tool": [0, 0, -1]},
        scene_refs={"start": {"position_world": [0, 0, 0.6]},
                    "end": {"position_world": [0.1, 0.2, 0.6]}, "direction": [0, 0, -1],
                    "motion_path": [{"position_world": p} for p in ([0, 0, 0.6], [0.1, 0, 0.6], [0.1, 0.2, 0.6])]},
        scene_motion_path_ref="scene.motion_path").to_json()
    inputs = build_mainline_inputs_from_image_grounded_program(program)
    assert inputs["derived_scene_geometry"]["end_point"] == [0.1, 0.2, 0.6]
    axes = [c for c in inputs["program_ir"]["constraints"] if c["stage"].startswith("action") and c["type"] == "align_axis"]
    assert len(axes) == 2
    assert all(c["refs"]["target_axis"] == "target.axes.direction" for c in axes)


def observation(tmp_path):
    from PIL import Image
    evidence = tmp_path / "observation"
    evidence.mkdir()
    Image.new("RGB", (16, 16), "white").save(evidence / "rgb.png")
    np.save(evidence / "depth.npy", np.ones((16, 16)))
    write_json(evidence / "camera.json", {"width": 16, "height": 16, "fx": 30, "fy": 30, "cx": 8, "cy": 8,
        "camera_to_world": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.56], [0, 0, 0, 1]]})
    write_json(evidence / "scene_evidence.json", {"items": [
        {"id": "view_rgb", "kind": "scene_rgb", "path": "rgb.png"},
        {"id": "view_depth", "kind": "scene_depth", "path": "depth.npy"},
        {"id": "view_camera", "kind": "camera", "path": "camera.json"}],
        "metadata": {"prompt_images": ["rgb.png"], "grounding_image_index": 0}})
    keypoints = {"start_point": {"u": 8, "v": 8}, "end_point": None, "path_points": [],
                 "direction_decision": {"selected_default_axis_id": "opposite_support_surface_normal"}}
    task = read_json(ROOT / "examples/mallet_hammer_strike.json")
    refs = read_json(ROOT / "assets/tool_refs/mallet_hammer.json")
    return evidence / "scene_evidence.json", keypoints, task, refs


def test_grounding_view_must_have_its_own_depth_and_calibration(tmp_path):
    evidence, points, task, refs = observation(tmp_path)
    data = read_json(evidence)
    data["items"][2]["id"] = "other_view_camera"
    write_json(evidence, data)
    with pytest.raises(ValueError, match="paired camera"):
        build_program(task=task, evidence_path=evidence, keypoints=points, tool_refs=refs, approach_offset_m=0.03)


def test_entry_plane_keeps_endpoint_depth(tmp_path):
    evidence, points, task, refs = observation(tmp_path)
    task["interaction_plane"] = {"point_world": [0, 0, 0.7], "normal_world": [0, 0, 1],
                                  "point_roles": ["start"], "source": "provided_scene_geometry"}
    points["end_point"] = {"u": 8, "v": 8}
    program = build_program(task=task, evidence_path=evidence, keypoints=points, tool_refs=refs, approach_offset_m=0.03)
    assert program["scene_refs"]["start"]["position_world"][2] == pytest.approx(0.7)
    assert program["scene_refs"]["end"]["position_world"][2] == pytest.approx(0.56)


def test_cli_compiles_from_another_working_directory(tmp_path):
    evidence, points, task, refs = observation(tmp_path)
    write_json(tmp_path / "task.json", task)
    write_json(tmp_path / "keypoints.json", points)
    write_json(tmp_path / "tool.json", refs)
    output = tmp_path / "plan"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, str(ROOT / "run.py"), "plan",
        "--task", str(tmp_path / "task.json"), "--output", str(output),
        "--scene-evidence", str(evidence), "--scene-keypoints", str(tmp_path / "keypoints.json"),
        "--tool-refs", str(tmp_path / "tool.json")], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    goals = read_json(output / "fixed_goals.json")
    assert goals["goals"] and all(len(pose) == 7 for pose in goals["goals"])
    subgoals = read_json(output / "tool_pose_subgoals.json")
    assert subgoals["goals"][-1]["contact_point_goal"]["position_world"] == pytest.approx([0, 0, 0.56])


def test_execution_uses_only_backend_goal_progress_and_shared_control_settings():
    task = read_json(ROOT / "examples/mallet_hammer_strike.json")
    settings = read_json(ROOT / "settings.json")
    goals = {"start_pose": task["execution"]["simtoolreal_start_pose"], "goals": [[0, 0, 0.6, 0, 0, 0, 1]]}
    overrides = environment_overrides(task, goals, settings)
    assert overrides["task.env.fixedGoalStates"] == goals["goals"]
    assert overrides["task.env.goalTolerance"] == settings["goal_tolerance_m"]
    assert not overrides["task.env.use_hack_object_pos_offset"]


def test_selected_refs_do_not_expose_optimizer_alternatives():
    refs = {"contact_point_tool": [0, 0, 0], "motion_axis_tool": [0, 0, 1],
            "motion_axis_candidates_tool": [{"vector_tool": [1, 0, 0]}],
            "surface_geometry_tool": {"motion_axis_candidates_tool": [{"vector_tool": [1, 0, 0]}]}}
    result = compact_tool_refs(refs, "unit")
    assert "motion_axis_candidates_tool" not in result
    assert "motion_axis_candidates_tool" not in result["surface_geometry_tool"]


def test_cached_refs_must_belong_to_the_selected_tool():
    refs = {"tool_id": "other", "contact_point_tool": [0, 0, 0], "motion_axis_tool": [0, 0, 1]}
    with pytest.raises(ValueError, match="belong to"):
        compact_tool_refs(refs, "unit")


@pytest.mark.parametrize("key,value", [("contact_point_tool", [float("nan"), 0, 0]),
                                     ("motion_axis_tool", [0, 0, 0]),
                                     ("motion_axis_tool", [1, 0])])
def test_selected_tool_geometry_must_be_finite_and_nonzero(key, value):
    refs = {"contact_point_tool": [0, 0, 0], "motion_axis_tool": [0, 0, 1], key: value}
    with pytest.raises(ValueError):
        compact_tool_refs(refs, "unit")


def test_reused_observation_uses_current_task_instruction(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ToGo.cli.stages import run_vlm_scene_keypoint_grounding as grounding
    evidence, _, _, _ = observation(tmp_path)
    write_json(evidence, {**read_json(evidence), "instruction": "Old task"})
    class Client:
        def generate(self, **kwargs):
            assert kwargs["task_instruction"] == "Current task"
            assert "Current task" in kwargs["prompt"]
            return SimpleNamespace(raw_text=json.dumps({"start_point": {"u": 8, "v": 8},
                "action_direction": {"start": {"u": 8, "v": 8}, "end": {"u": 9, "v": 8}},
                "end_point": {"u": 8, "v": 8}, "path_points": [], "direction_decision": {
                    "source": "default_axis_candidate", "selected_default_axis_id": "opposite_support_surface_normal"}}),
                metadata={})
    monkeypatch.setattr(grounding, "build_vlm_client", lambda **kwargs: Client())
    output = tmp_path / "grounding"
    monkeypatch.setattr(sys, "argv", ["grounding", "--scene-evidence", str(evidence),
        "--output-dir", str(output), "--instruction", "Current task"])
    assert grounding.main() == 0
    points = read_json(output / "scene_keypoints.json")
    assert points["instruction"] == "Current task"
    assert "raw_response" not in points
    assert "vlm_metadata" not in points
