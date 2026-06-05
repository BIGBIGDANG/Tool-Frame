import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_rollout_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "parttoolgen" / "execution" / "simtoolreal_rollout.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("parttoolgen_simtoolreal_rollout", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rollout = _load_rollout_module()


def _subgoals(stage: str) -> dict:
    return {
        "goals": [
            {
                "stage": stage,
                "pose_world_tool": {
                    "position": [0.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "expected_contact": {
                    "tool_part": "candidate.points.tip",
                    "scene_target": "scene.keypoints.target",
                },
                "contact_point_goal": {
                    "part_ref": "candidate.points.tip",
                    "position_world": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ]
    }


def _scene() -> dict:
    return {
        "keypoints": {
            "target": {
                "xyz_world": [5.0, 0.0, 0.0],
            }
        }
    }


def test_fixed_goal_start_pose_map_is_derived_from_artifact() -> None:
    fixed_goals = {
        "start_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "goals": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    }
    tool_pose_subgoals = {"goals": [{"stage": "contact"}]}

    goal_index_map = rollout._derive_goal_index_map(fixed_goals, tool_pose_subgoals)

    assert goal_index_map[0]["stage"] == "start_hold"
    assert goal_index_map[0]["source_goal_index"] is None
    assert goal_index_map[1]["source_goal_index"] == 0


def test_start_hold_does_not_inherit_first_subgoal_semantic_metadata() -> None:
    fixed_goals = {
        "start_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "goals": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    }
    goal_index_map = rollout._derive_goal_index_map(fixed_goals, {"goals": [_subgoals("contact")["goals"][0]]})

    spec = rollout._functional_observation_spec(
        tool_pose_subgoals=_subgoals("contact"),
        goal_index=0,
        scene_graph=_scene(),
        goal_index_map=goal_index_map,
    )

    assert spec is None


def test_transition_semantic_check_uses_planned_functional_waypoint() -> None:
    spec = rollout._functional_observation_spec(
        tool_pose_subgoals=_subgoals("approach_transition_1"),
        goal_index=0,
        scene_graph=_scene(),
    )
    observation = rollout._functional_observation_from_pose(
        actual_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        spec=spec,
        max_contact_point_target_error_m=0.03,
    )

    assert spec["semantic_gate"] is False
    assert observation["semantic_gate"] is False
    assert observation["contact_point_target_source"] == "planned_contact_point_goal"
    assert observation["functional_target_error_m"] == 0.0
    assert "semantic_contact_check" not in observation


def test_generated_transition_waypoint_keeps_tracking_but_not_semantic_gate() -> None:
    spec = rollout._functional_observation_spec(
        tool_pose_subgoals=_subgoals("approach_transition_1"),
        goal_index=0,
        scene_graph=_scene(),
    )
    observation = rollout._functional_observation_from_pose(
        actual_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        spec=spec,
        max_contact_point_target_error_m=0.03,
    )

    assert spec["semantic_gate"] is False
    assert observation["semantic_gate"] is False
    assert observation["contact_point_target_source"] == "planned_contact_point_goal"
    assert observation["functional_target_error_m"] == 0.0
    assert "semantic_contact_check" not in observation


def test_named_semantic_waypoint_still_uses_scene_target_ref() -> None:
    spec = rollout._functional_observation_spec(
        tool_pose_subgoals=_subgoals("action_waypoint_4"),
        goal_index=0,
        scene_graph=_scene(),
    )
    observation = rollout._functional_observation_from_pose(
        actual_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        spec=spec,
        max_contact_point_target_error_m=0.03,
    )

    assert spec["semantic_gate"] is True
    assert observation["contact_point_target_source"] == "scene_ref"
    assert observation["functional_target_error_m"] == 4.0


def test_terminal_semantic_check_uses_scene_target_ref() -> None:
    spec = rollout._functional_observation_spec(
        tool_pose_subgoals=_subgoals("action_terminal"),
        goal_index=0,
        scene_graph=_scene(),
    )
    observation = rollout._functional_observation_from_pose(
        actual_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        spec=spec,
        max_contact_point_target_error_m=0.03,
    )

    assert observation["contact_point_target_source"] == "scene_ref"
    assert observation["functional_target_error_m"] == 4.0


def test_default_eval_success_tolerance_is_limited_by_semantic_threshold() -> None:
    resolved = rollout._resolve_eval_success_tolerance_m(
        explicit_eval_success_tolerance_m=None,
        max_tracking_error_m=0.035,
        max_functional_target_error_m=0.03,
    )

    assert resolved["source"] == "parttoolgen_semantic_threshold"
    assert resolved["value_m"] == pytest.approx(0.018)


def test_explicit_eval_success_tolerance_overrides_semantic_default() -> None:
    resolved = rollout._resolve_eval_success_tolerance_m(
        explicit_eval_success_tolerance_m=0.012,
        max_tracking_error_m=0.035,
        max_functional_target_error_m=0.03,
    )

    assert resolved["source"] == "explicit_cli_or_api"
    assert resolved["value_m"] == pytest.approx(0.012)


def test_camera_capture_restores_viewer_sync(monkeypatch) -> None:
    class FakeImage:
        size = 4

        def reshape(self, *_shape):
            return self

        def copy(self):
            return "frame"

    class FakeGym:
        def __init__(self) -> None:
            self.rendered = False

        def render_all_camera_sensors(self, _sim) -> None:
            self.rendered = True

        def get_camera_image(self, _sim, _env, _camera_handle, _image_kind):
            return FakeImage()

    class FakeEnv:
        def __init__(self) -> None:
            self.enable_viewer_sync = False
            self.gym = FakeGym()
            self.sim = object()
            self.envs = [object()]
            self.camera_handle = object()
            self.index_to_view = 0
            self.camera_properties = types.SimpleNamespace(width=1, height=1)

        def render(self) -> None:
            assert self.enable_viewer_sync is True

    monkeypatch.setitem(
        sys.modules,
        "isaacgym",
        types.SimpleNamespace(gymapi=types.SimpleNamespace(IMAGE_COLOR=0)),
    )
    env = FakeEnv()

    frame = rollout._capture_camera_frame(env)

    assert frame == "frame"
    assert env.gym.rendered is True
    assert env.enable_viewer_sync is False
