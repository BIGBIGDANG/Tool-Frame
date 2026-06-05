import json

from parttoolgen.program.part_semantic_labels import (
    augment_visual_evidence_with_part_semantics,
    build_from_sampart_semantic_map,
    merge_part_semantic_labels,
    parse_part_semantic_label_response,
    render_part_semantic_label_prompt,
)
from parttoolgen.program.prompting import build_program_symbol_table


def _grounding() -> dict:
    return {
        "confidence": 0.8,
        "source": "unit_test",
        "provenance": {"observation_id": "obs", "method": "unit_test"},
    }


def _tool_part_graph() -> dict:
    return {
        "schema_version": "v0.1",
        "tool_id": "mallet_hammer",
        "tool_frame": "tool",
        "parts": {
            "part_002": {
                "label": "sampart3d_segment_2",
                "role_candidates": ["tool_part_candidate"],
                "mask_ref": "sampart3d:face_label:2",
                "point_count": 10,
                "centroid_xyz_tool": [0.0, 0.0, 0.0],
                "obb": {
                    "center_xyz_tool": [0.0, 0.0, 0.0],
                    "extent_xyz": [0.1, 0.2, 0.3],
                    "axes_tool": {
                        "x": [1.0, 0.0, 0.0],
                        "y": [0.0, 1.0, 0.0],
                        "z": [0.0, 0.0, 1.0],
                    },
                },
                **_grounding(),
            }
        },
        "relations": [
            {
                "type": "part_of",
                "subject": "tool.parts.part_002",
                "object": "tool.mallet_hammer",
            }
        ],
        "metadata": {
            "created_by": "unit_test",
            "geometry_mode": "prior",
            "segmentation_mode": "sampart3d",
        },
    }


def _grounding_candidates() -> dict:
    return {
        "schema_version": "v0.1",
        "grounding_id": "g",
        "observation_id": "obs",
        "frames": {"tool": {"frame_type": "tool"}},
        "marks": {},
        "points": {
            "part_002_surface_center": {
                "label": "part_002_surface_center",
                "candidate_type": "geometry_surface_center",
                "part_ref": "tool.parts.part_002",
                "xyz_tool": [0.0, 0.0, 0.0],
                **_grounding(),
            }
        },
        "regions": {},
        "axes": {},
        "surfaces": {},
        "parts": {
            "part_002": {
                "label": "sampart3d_segment_2",
                "candidate_type": "sampart3d_part",
                "part_ref": "tool.parts.part_002",
                "role_candidates": ["tool_part_candidate"],
                **_grounding(),
            }
        },
        "metadata": {"created_by": "unit_test", "grounding_mode": "mixed"},
    }


def _scene() -> dict:
    return {
        "schema_version": "v0.1",
        "scene_id": "scene",
        "frames": {"world": {"type": "world"}},
        "objects": {},
        "keypoints": {},
        "axes": {},
        "regions": {},
        "surfaces": {},
        "relations": [],
        "metadata": {"created_by": "unit_test", "geometry_mode": "observed"},
    }


def _functional_tool_parts() -> dict:
    return {
        "schema_version": "v0.1",
        "tool_id": "mallet_hammer",
        "tool_frame": "tool",
        "functional_parts": {
            "tip": {
                "role": "contact_patch",
                "center": {"xyz_tool": [0.0, 0.0, 0.0]},
                "normal": {"vector_tool": [0.0, 0.0, 1.0]},
                "axis": {"vector_tool": [1.0, 0.0, 0.0]},
                "frame": {
                    "origin": [0.0, 0.0, 0.0],
                    "x_axis": [1.0, 0.0, 0.0],
                    "y_axis": [0.0, 1.0, 0.0],
                    "z_axis": [0.0, 0.0, 1.0],
                },
                "T_tool_functional": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                **_grounding(),
            }
        },
        "grasp_references": {
            "grasp": {
                "position_tool": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                **_grounding(),
            }
        },
        "metadata": {"created_by": "unit_test", "geometry_mode": "prior"},
    }


def test_part_semantic_labels_merge_into_prompt_indices() -> None:
    raw = json.dumps(
        {
            "schema_version": "v0.1",
            "label_id": "labels",
            "tool_id": "mallet_hammer",
            "parts": {
                "part_002": {
                    "part_ref": "tool.parts.part_002",
                    "semantic_label": "mallet head",
                    "aliases": ["mallet head", "striking face"],
                    "visual_description": "Large block-like head region.",
                    "confidence": 0.75,
                    "evidence_items": ["mallet_overlay"],
                    "source": "vlm_part_semantic_labeler",
                    "provenance": {"method": "vlm_part_semantic_alias", "observation_id": "obs"},
                }
            },
            "metadata": {"created_by": "unit_test", "mode": "vlm_part_semantic_aliases"},
        }
    )
    labels = parse_part_semantic_label_response(
        raw,
        tool_part_graph=_tool_part_graph(),
        label_id="labels",
    )
    merged_tool, merged_candidates = merge_part_semantic_labels(
        tool_part_graph=_tool_part_graph(),
        semantic_labels=labels,
        grounding_candidates=_grounding_candidates(),
    )

    assert merged_tool["parts"]["part_002"]["label"] == "mallet head"
    assert "semantic_alias:striking face" in merged_tool["parts"]["part_002"]["role_candidates"]
    assert merged_candidates["points"]["part_002_surface_center"]["part_semantic_label"] == "mallet head"

    symbol_table = build_program_symbol_table(
        _scene(),
        _functional_tool_parts(),
        grounding_candidates=merged_candidates,
        tool_part_graph=merged_tool,
    )
    assert "tool.parts.part_002" in symbol_table["role_index"]["tool_parts"]["semantic_alias:striking face"]
    assert (
        "candidate.parts.part_002"
        in symbol_table["role_index"]["candidate_parts"]["semantic_alias:striking face"]
    )


def test_part_semantic_prompt_and_visual_rows_keep_aliases_non_refs() -> None:
    visual = {
        "schema_version": "v0.1",
        "evidence_id": "e",
        "items": [
            {
                "id": "mallet_overlay",
                "kind": "mask_overlay",
                "media_type": "image",
                "path": "overlay.png",
                "allowed_use": "planning_context",
            }
        ],
        "metadata": {
            "anchor_ref_table": [
                {
                    "tool_id": "mallet_hammer",
                    "number": 4,
                    "part_id": "part_002",
                    "part_ref": "tool.parts.part_002",
                    "point_ref": "candidate.points.part_002_surface_center",
                }
            ]
        },
    }
    prompt = render_part_semantic_label_prompt(
        tool_part_graph=_tool_part_graph(),
        visual_evidence=visual,
        label_id="labels",
    )
    assert "not executable refs" in prompt
    labels = parse_part_semantic_label_response(
        json.dumps(
            {
                "schema_version": "v0.1",
                "label_id": "labels",
                "tool_id": "mallet_hammer",
                "parts": {
                    "part_002": {
                        "part_ref": "tool.parts.part_002",
                        "semantic_label": "mallet head",
                        "aliases": ["mallet head", "striking face"],
                        "visual_description": "Large block-like head region.",
                        "confidence": 0.75,
                        "evidence_items": ["mallet_overlay"],
                        "source": "vlm_part_semantic_labeler",
                        "provenance": {"method": "vlm_part_semantic_alias", "observation_id": "obs"},
                    }
                },
                "metadata": {"created_by": "unit_test", "mode": "vlm_part_semantic_aliases"},
            }
        ),
        tool_part_graph=_tool_part_graph(),
        label_id="labels",
    )
    augmented = augment_visual_evidence_with_part_semantics(
        visual_evidence=visual,
        semantic_labels=labels,
    )
    row = augmented["metadata"]["anchor_ref_table"][0]
    assert row["part_semantic_label"] == "mallet head"
    assert "striking face" in row["part_semantic_aliases"]


def test_sampart_native_semantic_map_is_preferred_over_captioning() -> None:
    labels = build_from_sampart_semantic_map(
        tool_part_graph=_tool_part_graph(),
        semantic_map={"mallet_hammer": ["handle", "neck", "mallet head"]},
        label_id="native_labels",
        source_artifact="sampart_semantic.json",
    )

    assert labels["metadata"]["mode"] == "provider_supplied_part_semantic_aliases"
    assert labels["parts"]["part_002"]["semantic_label"] == "mallet head"
    assert labels["parts"]["part_002"]["source"] == "sampart3d_semantic_labels"
