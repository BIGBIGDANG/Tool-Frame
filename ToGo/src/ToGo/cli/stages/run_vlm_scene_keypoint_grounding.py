#!/usr/bin/env python3
"""Ask a VLM to mark image-space scene keypoints for ToolFrame."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
SRC = ROOT / "ToGo" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ToGo.constraints.vlm_client import build_vlm_client  # noqa: E402

END_POINT_APPLICABILITY_VALUES = {"required", "optional", "not_applicable", "uncertain"}
DIRECTION_SOURCE_VALUES = {"visible_action_direction", "default_axis_candidate"}
DEFAULT_AXIS_IDS = {"support_surface_normal", "opposite_support_surface_normal"}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _strip_json_response(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _prompt(
    *,
    instruction: str,
    width: int,
    height: int,
    num_images: int,
    grounding_image_index: int,
    grounding_image_path: Path,
    grounding_frame_convention: Mapping[str, Any] | None = None,
    public_scene_context: Mapping[str, Any] | None = None,
    request_world_action_axis: bool = False,
) -> str:
    # Legacy callers may still pass request_world_action_axis; scene VLM
    # grounding now uses either visible 2D scene references or an explicit
    # public default-axis candidate selected by the VLM.
    _ = request_world_action_axis
    image_note = (
        f"You are given {num_images} images. Use all images for visual context, "
        f"but return pixel coordinates only in image #{grounding_image_index + 1}: "
        f"{grounding_image_path.name}."
        if num_images > 1
        else "You are given one calibrated simulation image."
    )
    frame_note = ""
    if grounding_frame_convention:
        frame_note = (
            "\nGrounding image frame convention:\n"
            f"- image up: {grounding_frame_convention.get('image_up')}\n"
            f"- image down: {grounding_frame_convention.get('image_down')}\n"
            f"- image right: {grounding_frame_convention.get('image_right')}\n"
            "- Draw action_direction arrows in this image frame.\n"
        )
    default_axis_note = ""
    default_axis_schema = ""
    if isinstance(public_scene_context, Mapping) and public_scene_context.get("default_axis_candidates"):
        default_axis_note = (
            "\nPublic default direction-axis candidates are available for cases "
            "where the task acts along a support-surface normal and no reliable "
            "visible 2D direction exists. These candidates are task/world priors, "
            "not hidden robot motion plans:\n"
            f"{json.dumps(public_scene_context.get('default_axis_candidates'), indent=2)}\n"
            "Use a default-axis candidate only when the task semantics require "
            "motion into/away from the support surface and a visible action "
            "direction would be misleading. This can be true even when both "
            "start_point and end_point are visible, such as entering a visible "
            "opening where the 2D points mark the mouth and interior but the "
            "physical use axis is depth/normal. Otherwise choose "
            "visible_action_direction."
        )
        default_axis_schema = (
            ',\n  "direction_decision": {\n'
            '    "source": "visible_action_direction|default_axis_candidate",\n'
            '    "selected_default_axis_id": "support_surface_normal|opposite_support_surface_normal|null",\n'
            '    "confidence": <0-1>,\n'
            '    "reason": "<why visible 2D direction or the selected default axis is appropriate>"\n'
            "  }"
        )
    return f"""
You are the scene-grounding module for a tool-use compiler.
{image_note}
{frame_note}
Task:
{instruction}

Grounding image size: width={width}, height={height}.

Return a sparse Scene Reference Program: task-level scene references that
state what the tool should act on, not how the robot should move. The compiler
will lift these image references with RGB-D and generate metric axes, approach
offsets, contact poses, subgoals, and executable trajectories downstream.

Scene-reference interface:
1. target_object: the visible object or support region selected by the task.
2. start_point: the visible point where functional tool contact begins.
3. end_point: the intended visible outcome point, if a distinct endpoint is
   semantically needed or visible.
4. path_points: an optional sparse ordered list of visible contact-stroke
   waypoints between start_point and end_point. Use path_points only when the
   intended functional contact path is non-straight or needs sparse
   intermediate contact waypoints. Use the fewest path_points necessary; for
   a straight contact stroke, return an empty path_points list.
5. action_direction: a compact scene-level outcome cue from start_point toward
   the intended use outcome. It is not a robot trajectory.
6. end_point_decision: whether a distinct endpoint is required, optional,
   not applicable, or uncertain.
7. direction_decision: when default direction-axis candidates are provided,
   choose whether the task direction should come from visible 2D scene geometry
   or from a public default axis.

Grounding principles:
- Use only visible, task-relevant pixels in the specified grounding image.
- Points may lie on the selected target object, on its usable support surface,
  on a visible boundary to be crossed, or in the adjacent visible outcome area
  when the task asks to move an object out of or across a region.
- For surface-use tasks such as wiping, drawing, erasing, pressing, or marking,
  place start_point, end_point, and any path_points on the intended contact
  stroke of the tool's functional contact point on the target surface.
- Do not use path_points to describe target-region corners, bounding-box
  corners, or coverage-region vertices unless the tool's functional contact
  point should physically pass through those points in order.
- Do not outline the full target region with path_points. Prefer the minimum
  ordered stroke representation that a tool contact point should follow.
- For object-displacement tasks such as pushing, sweeping, or sliding an object,
  start_point is where the tool first acts on the object; end_point or
  action_direction.end marks the intended final scene reference, crossed
  boundary, or visible outcome direction.
- For insertion or entry tasks, start_point is the visible entry/mouth point;
  end_point is the deepest visible interior point if visible. If the final
  depth is not visible, repeat start_point and use direction_decision to encode
  the task axis.
- If no separate endpoint is visible or semantically needed, set end_point and
  action_direction.end equal to start_point.
- Keep the program sparse. Do not add interpolation samples, robot approach
  poses, controller hints, or hidden 3D waypoints.
- If multiple views are provided, use all views for context but return
  coordinates only in the specified grounding image.
{default_axis_note}

Return 2D points in [y, x] format normalized to 0-1000, where y is vertical
image coordinate and x is horizontal image coordinate.

Return exactly one JSON object:
{{
  "target_object": "short visible target identifier",
  "start_point": {{
    "point": [<y>, <x>],
    "confidence": <0-1>,
    "reason": "why this is where functional contact begins"
  }},
  "action_direction": {{
    "start": {{
      "point": [<y>, <x>],
      "confidence": <0-1>,
      "reason": "why this is the direction start"
    }},
    "end": {{
      "point": [<y>, <x>],
      "confidence": <0-1>,
      "reason": "why this is the direction end"
    }},
    "confidence": <0-1>,
    "reason": "why this compact scene-level direction represents the task outcome"
  }},
  "path_points": [
    {{
      "point": [<y>, <x>],
      "label": "optional sparse scene-reference label",
      "confidence": <0-1>,
      "reason": "why this ordered contact-stroke waypoint is physically necessary"
    }}
  ],
  "end_point": {{
    "point": [<y>, <x>],
    "confidence": <0-1>,
    "reason": "why this is the final visible outcome reference"
  }},
  "end_point_decision": {{
    "applicability": "required|optional|not_applicable|uncertain",
    "confidence": <0-1>,
    "reason": "why a separate endpoint is or is not needed"
  }}{default_axis_schema}
}}
""".strip()


def _validate_point(name: str, value: Any, *, width: int, height: int) -> dict[str, float]:
    if isinstance(value, list) and len(value) == 2:
        value = {"point": value}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    normalized_yx = value.get("point")
    if normalized_yx is None:
        normalized_yx = value.get("point_2d")
    if isinstance(normalized_yx, list) and len(normalized_yx) == 2:
        y_norm = float(normalized_yx[0])
        x_norm = float(normalized_yx[1])
        if not (0.0 <= x_norm <= 1000.0 and 0.0 <= y_norm <= 1000.0):
            raise ValueError(f"{name} normalized point {normalized_yx} outside 0-1000")
        u = x_norm / 1000.0 * float(width)
        v = y_norm / 1000.0 * float(height)
    else:
        # Backwards-compatible path for older local responses.
        u = float(value["u"])
        v = float(value["v"])
        x_norm = u / float(width) * 1000.0
        y_norm = v / float(height) * 1000.0
    clipped = False
    if not (0.0 <= u < width and 0.0 <= v < height):
        if -1.0 <= u <= float(width) and -1.0 <= v <= float(height):
            u = min(max(u, 0.0), float(width - 1))
            v = min(max(v, 0.0), float(height - 1))
            clipped = True
        else:
            raise ValueError(f"{name} pixel ({u}, {v}) outside image {width}x{height}")
    confidence = value.get("confidence")
    if confidence is None:
        confidence = 1.0
    confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "u": u,
        "v": v,
        "confidence": confidence,
        "point_yx_1000": [y_norm, x_norm],
        "coordinate_format": "yx_normalized_0_1000",
        "pixel_clipped_to_image_bounds": clipped,
    }


def _validate_optional_point(
    name: str,
    value: Any,
    *,
    width: int,
    height: int,
) -> dict[str, float] | None:
    if value is None:
        return None
    return _validate_point(name, value, width=width, height=height)


def _validate_path_points(
    value: Any,
    *,
    width: int,
    height: int,
    field_name: str = "path_points",
) -> list[dict[str, float | str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    path_points: list[dict[str, float | str]] = []
    for index, row in enumerate(value):
        point = _validate_point(f"{field_name}[{index}]", row, width=width, height=height)
        path_points.append(
            {
                "id": (
                    str(row.get("id") or f"path{index}")
                    if isinstance(row, Mapping)
                    else f"path{index}"
                ),
                **point,
            }
        )
    return path_points


def _validate_target_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "label": "target object",
            "description": "not specified by VLM response",
            "confidence": 0.0,
        }
    confidence = value.get("confidence", 1.0)
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_f = 0.0
    return {
        "label": str(value.get("label") or value.get("name") or "target object"),
        "description": str(value.get("description") or value.get("reason") or ""),
        "confidence": confidence_f,
    }


def _validate_decision(
    name: str,
    value: Any,
    *,
    allowed: set[str],
    default_applicability: str,
    default_reason: str,
) -> dict[str, Any]:
    if value is None:
        return {
            "applicability": default_applicability,
            "confidence": 0.0,
            "reason": default_reason,
            "source": "parser_default_missing_vlm_decision",
        }
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    applicability = str(value.get("applicability") or "").strip().lower()
    applicability = applicability.replace("-", "_").replace(" ", "_")
    if applicability not in allowed:
        raise ValueError(
            f"{name}.applicability must be one of {sorted(allowed)}; got {applicability!r}"
        )
    confidence = value.get("confidence", 1.0)
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_f = 0.0
    return {
        "applicability": applicability,
        "confidence": confidence_f,
        "reason": str(value.get("reason") or ""),
        "source": "vlm_task_semantic_grounding",
    }


def _validate_direction_decision(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "source": "visible_action_direction",
            "selected_default_axis_id": None,
            "confidence": 1.0,
            "reason": "No default-axis decision was returned; using visible action_direction.",
            "source_defaulted": True,
        }
    if not isinstance(value, Mapping):
        raise ValueError("direction_decision must be an object")
    source = str(value.get("source") or "").strip().lower()
    source = source.replace("-", "_").replace(" ", "_")
    if source not in DIRECTION_SOURCE_VALUES:
        raise ValueError(
            "direction_decision.source must be one of "
            f"{sorted(DIRECTION_SOURCE_VALUES)}"
        )
    selected = value.get("selected_default_axis_id")
    if selected in {"", "null"}:
        selected = None
    if source == "default_axis_candidate":
        if not isinstance(selected, str) or selected not in DEFAULT_AXIS_IDS:
            raise ValueError(
                "direction_decision.selected_default_axis_id must be one of "
                f"{sorted(DEFAULT_AXIS_IDS)} when source is default_axis_candidate"
            )
    elif selected not in {None, ""}:
        raise ValueError(
            "direction_decision.selected_default_axis_id must be null unless "
            "source is default_axis_candidate"
        )
    confidence = value.get("confidence", 1.0)
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_f = 0.0
    return {
        "source": source,
        "selected_default_axis_id": selected,
        "confidence": confidence_f,
        "reason": str(value.get("reason") or ""),
    }


def _parse_grounding_response(data: Mapping[str, Any], *, width: int, height: int) -> dict[str, Any]:
    target_object = _validate_target_object(data.get("target_object"))
    start_point = _validate_point("start_point", data.get("start_point"), width=width, height=height)
    action_direction_raw = data.get("action_direction")
    if not isinstance(action_direction_raw, Mapping):
        raise ValueError("action_direction must be an object")
    action_direction = {
        "start": _validate_point(
            "action_direction.start",
            action_direction_raw.get("start"),
            width=width,
            height=height,
        ),
        "end": _validate_point(
            "action_direction.end",
            action_direction_raw.get("end"),
            width=width,
            height=height,
        ),
    }
    raw_end_point = _validate_optional_point(
        "end_point",
        data.get("end_point"),
        width=width,
        height=height,
    )
    end_point_decision = _validate_decision(
        "end_point_decision",
        data.get("end_point_decision"),
        allowed=END_POINT_APPLICABILITY_VALUES,
        default_applicability="required" if raw_end_point is not None else "not_applicable",
        default_reason="VLM response did not include end_point_decision; inferred from end_point.",
    )
    direction_decision = _validate_direction_decision(data.get("direction_decision"))
    raw_path_points = _validate_path_points(
        data.get("path_points"),
        width=width,
        height=height,
        field_name="path_points",
    )
    path_points_source = "vlm_path_points"
    if not raw_path_points and data.get("waypoints") is not None:
        raw_path_points = _validate_path_points(
            data.get("waypoints"),
            width=width,
            height=height,
            field_name="waypoints",
        )
        path_points_source = "legacy_vlm_waypoints_as_path_points"
    end_point = raw_end_point
    end_point_source = "vlm_end_point" if raw_end_point is not None else None
    path_points = list(raw_path_points)
    if path_points and end_point is None:
        end_point = dict(path_points[-1])
        end_point["source"] = "final_path_point_as_end_point"
        end_point_source = "final_path_point"
        end_point_decision = {
            "applicability": "required",
            "confidence": max(float(end_point_decision.get("confidence", 0.0)), float(end_point.get("confidence", 0.0))),
            "reason": (
                "The ordered path_points field includes a final endpoint; the "
                "parser uses the final path point as end_point while preserving "
                "the complete returned path."
            ),
            "source": "parser_final_path_point_as_end_point",
        }
    if end_point is None:
        raise ValueError(
            "end_point is required; for contact-only tasks return end_point equal to start_point"
        )
    intermediate_path_points = path_points[1:-1] if len(path_points) >= 2 else []
    return {
        "target_object": target_object,
        "start_point": start_point,
        "end_point_decision": end_point_decision,
        "direction_decision": direction_decision,
        "action_direction": action_direction,
        "raw_end_point": raw_end_point,
        "raw_path_points": raw_path_points,
        "path_points_source": path_points_source,
        "end_point_source": end_point_source,
        "end_point": end_point,
        "path_points": path_points,
        "intermediate_path_points": intermediate_path_points,
    }


def _semantic_refs(
    *,
    target_object: Mapping[str, Any],
    start_point: Mapping[str, Any],
    action_direction: Mapping[str, Any],
    end_point: Mapping[str, Any] | None,
    end_point_decision: Mapping[str, Any],
    direction_decision: Mapping[str, Any],
    path_points: list[dict[str, float | str]],
) -> dict[str, Any]:
    """Return method-level start/end/direction refs represented by VLM marks."""

    return {
        "schema_version": "toolframe_semantic_refs_v3",
        "source": "vlm_task_conditioned_scene_grounding",
        "start": {
            "role": "functional_start",
            "target_object": dict(target_object),
            "point_ref": "start_point",
            "refs": [
                {
                    "id": "start.start_point",
                    "kind": "point_2d",
                    "source_key": "start_point",
                    "role": "where the tool first functionally acts on the target",
                    "value": dict(start_point),
                }
            ],
        },
        "use_stage": {
            "role": "post_start_use_reference",
            "direction_ref": "action_direction",
            "end_ref": "end_point" if end_point is not None else None,
            "end_point_decision": dict(end_point_decision),
            "direction_decision": dict(direction_decision),
            "refs": [
                {
                    "id": "use_stage.action_direction",
                    "kind": "axis_2d",
                    "source_key": "action_direction",
                    "role": "direction or visual anchor from use-stage point refs",
                    "value": dict(action_direction),
                },
                *(
                    [
                        {
                            "id": "use_stage.default_axis_candidate",
                            "kind": "axis_candidate",
                            "source_key": "direction_decision",
                            "role": "VLM-selected public support-surface default axis",
                            "value": dict(direction_decision),
                        }
                    ]
                    if direction_decision.get("source") == "default_axis_candidate"
                    else []
                ),
                *(
                    [
                        {
                            "id": "use_stage.end_point",
                            "kind": "point_2d",
                            "source_key": "end_point",
                            "role": "visible image endpoint for the executable use-stage path",
                            "value": dict(end_point),
                        }
                    ]
                    if end_point is not None
                    else []
                ),
            ],
            "path_ref": "path_points" if path_points else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--response-mode", default="openai-compatible")
    parser.add_argument("--model", default=None)
    parser.add_argument("--instruction", help="Current task instruction when reusing an observation")
    args = parser.parse_args()

    evidence = _load_json(args.scene_evidence)
    prompt_images = evidence.get("metadata", {}).get("prompt_images", [])
    if not prompt_images:
        raise FileNotFoundError("scene evidence metadata.prompt_images is empty")
    grounding_image_index = int(evidence.get("metadata", {}).get("grounding_image_index", 0))
    if not (0 <= grounding_image_index < len(prompt_images)):
        raise ValueError(
            f"metadata.grounding_image_index={grounding_image_index} outside prompt_images"
        )
    evidence_dir = Path(args.scene_evidence).resolve().parent
    def image_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else evidence_dir / path
    rgb_path = image_path(prompt_images[grounding_image_index])
    image_paths = [image_path(path) for path in prompt_images]
    from PIL import Image  # noqa: WPS433

    with Image.open(rgb_path) as image:
        width, height = image.size
    instruction = args.instruction if args.instruction is not None else str(evidence.get("instruction", ""))
    prompt = _prompt(
        instruction=instruction,
        width=width,
        height=height,
        num_images=len(image_paths),
        grounding_image_index=grounding_image_index,
        grounding_image_path=rgb_path,
        grounding_frame_convention=evidence.get("metadata", {}).get(
            "grounding_frame_convention"
        ),
        public_scene_context=evidence.get("metadata", {}).get("public_scene_context"),
        request_world_action_axis=False,
    )
    client = build_vlm_client(
        response_mode=args.response_mode,
        model=args.model,
        image_paths=image_paths,
        max_new_tokens=896,
    )
    response = client.generate(
        prompt=prompt,
        task_instruction=instruction,
        context_id=str(evidence.get("task_id") or "scene_keypoint_grounding"),
    )
    data = json.loads(_strip_json_response(response.raw_text))
    parsed = _parse_grounding_response(data, width=width, height=height)
    output = {
        "schema_version": "toolframe_scene_keypoint_grounding_v3",
        "task_id": evidence.get("task_id"),
        "instruction": instruction,
        "scene_evidence": str(Path(args.scene_evidence)),
        "target_object": parsed["target_object"],
        "start_point": parsed["start_point"],
        "end_point_decision": parsed["end_point_decision"],
        "direction_decision": parsed["direction_decision"],
        "action_direction": parsed["action_direction"],
        "raw_end_point": parsed["raw_end_point"],
        "raw_path_points": parsed["raw_path_points"],
        "path_points_source": parsed["path_points_source"],
        "end_point_source": parsed["end_point_source"],
        "end_point": parsed["end_point"],
        "path_points": parsed["path_points"],
        "intermediate_path_points": parsed["intermediate_path_points"],
        "semantic_refs": _semantic_refs(
            target_object=parsed["target_object"],
            start_point=parsed["start_point"],
            action_direction=parsed["action_direction"],
            end_point=parsed["end_point"],
            end_point_decision=parsed["end_point_decision"],
            direction_decision=parsed["direction_decision"],
            path_points=parsed["path_points"],
        ),
        "reason": data.get("reason", ""),
        "grounding_image_index": grounding_image_index,
        "grounding_image": str(rgb_path),
        "grounding_frame_convention": evidence.get("metadata", {}).get(
            "grounding_frame_convention"
        ),
        "public_scene_context": evidence.get("metadata", {}).get("public_scene_context"),
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "scene_keypoints.json", output)
    print(json.dumps({"scene_keypoints": str(output_dir / "scene_keypoints.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
