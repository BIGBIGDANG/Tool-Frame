from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_scene_grounding_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "ToGo" / "cli" / "stages" / "run_vlm_scene_keypoint_grounding.py"
    spec = importlib.util.spec_from_file_location("parttoolgen_scene_grounding", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scene_grounding = _load_scene_grounding_module()


def _response(
    *,
    path_points: list[dict[str, object]] | None = None,
    end_point: dict[str, object] | None = None,
    direction_decision: dict[str, object] | None = None,
) -> dict[str, object]:
    response = {
        "target_object": {
            "label": "target",
            "description": "visible target implied by the task",
            "confidence": 0.9,
        },
        "start_point": {"point": [500, 500], "confidence": 0.9},
        "end_point_decision": {
            "applicability": "required",
            "confidence": 0.8,
            "reason": "task-language decision",
        },
        "action_direction": {
            "start": {"point": [500, 500], "confidence": 0.9},
            "end": {"point": [450, 650], "confidence": 0.8},
        },
        "end_point": end_point or {"point": [450, 650], "confidence": 0.8},
        "path_points": path_points or [],
        "reason": "unit fixture",
    }
    if direction_decision is not None:
        response["direction_decision"] = direction_decision
    return response


def test_scene_prompt_asks_for_sparse_visible_scene_refs_not_trajectory() -> None:
    prompt = scene_grounding._prompt(
        instruction="Use the tool according to the visible task.",
        width=1024,
        height=768,
        num_images=1,
        grounding_image_index=0,
        grounding_image_path=Path("topdown_rgb.png"),
        public_scene_context={
            "default_axis_candidates": [
                {"id": "support_surface_normal"},
                {"id": "opposite_support_surface_normal"},
            ],
        },
        request_world_action_axis=False,
    )

    assert '"start_point"' in prompt
    assert '"end_point"' in prompt
    assert '"path_points"' in prompt
    assert '"action_direction"' in prompt
    assert "not how the robot should move" in prompt
    assert '"direction_decision"' in prompt


def test_scene_response_parser_returns_2d_scene_refs() -> None:
    parsed = scene_grounding._parse_grounding_response(
        _response(),
        width=1000,
        height=1000,
    )
    semantic_refs = scene_grounding._semantic_refs(
        target_object=parsed["target_object"],
        start_point=parsed["start_point"],
        action_direction=parsed["action_direction"],
        end_point=parsed["end_point"],
        end_point_decision=parsed["end_point_decision"],
        direction_decision=parsed["direction_decision"],
        path_points=parsed["path_points"],
    )

    assert parsed["end_point_decision"]["applicability"] == "required"
    assert parsed["direction_decision"]["source"] == "visible_action_direction"
    assert semantic_refs["start"]["point_ref"] == "start_point"
    assert semantic_refs["use_stage"]["direction_ref"] == "action_direction"
    assert semantic_refs["use_stage"]["end_ref"] == "end_point"


def test_scene_response_parser_uses_final_path_point_as_endpoint() -> None:
    response = _response(
        path_points=[
            {"id": "path0", "point": [500, 500], "confidence": 0.9},
            {"id": "path1", "point": [480, 560], "confidence": 0.7},
            {"id": "path2", "point": [450, 650], "confidence": 0.8},
        ],
    )
    response.pop("end_point")
    parsed = scene_grounding._parse_grounding_response(
        response,
        width=1000,
        height=1000,
    )

    assert parsed["end_point_source"] == "final_path_point"
    assert parsed["end_point"]["source"] == "final_path_point_as_end_point"
    assert len(parsed["path_points"]) == 3
    assert len(parsed["intermediate_path_points"]) == 1


def test_scene_response_parser_accepts_default_axis_candidate_decision() -> None:
    parsed = scene_grounding._parse_grounding_response(
        _response(
            direction_decision={
                "source": "default_axis_candidate",
                "selected_default_axis_id": "opposite_support_surface_normal",
                "confidence": 0.75,
                "reason": "The task moves into the support surface.",
            },
        ),
        width=1000,
        height=1000,
    )
    semantic_refs = scene_grounding._semantic_refs(
        target_object=parsed["target_object"],
        start_point=parsed["start_point"],
        action_direction=parsed["action_direction"],
        end_point=parsed["end_point"],
        end_point_decision=parsed["end_point_decision"],
        direction_decision=parsed["direction_decision"],
        path_points=parsed["path_points"],
    )

    assert parsed["direction_decision"]["source"] == "default_axis_candidate"
    assert parsed["direction_decision"]["selected_default_axis_id"] == "opposite_support_surface_normal"
    refs = semantic_refs["use_stage"]["refs"]
    assert any(row["id"] == "use_stage.default_axis_candidate" for row in refs)
