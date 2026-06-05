import json

from parttoolgen.checks.affordance_binding_sanity import check_affordance_binding_sanity
from parttoolgen.compiler.reference_resolver import ReferenceResolver
from parttoolgen.program.affordance_binding import (
    AffordanceBindingError,
    collect_binding_allowed_refs,
    parse_affordance_binding_response,
    render_affordance_binding_prompt,
)


def _prompt_symbol_table():
    return {
        "scene_id": "scene_smoke",
        "tool_id": "tool_smoke",
        "scene_symbols": {
            "objects": [],
            "keypoints": ["scene.keypoints.target_contact", "scene.keypoints.target_pre"],
            "axes": ["scene.axes.motion_axis"],
            "regions": [],
            "surfaces": [],
        },
        "tool_symbols": {
            "functional_parts": [
                "tool.functional_parts.part_000_contact.center",
                "tool.functional_parts.part_000_contact.normal",
                "tool.functional_parts.part_001_other.center",
            ],
            "grasp_references": ["tool.grasp_references.grasp_center"],
            "parts": ["tool.parts.part_000"],
        },
        "target_symbols": {"parts": [], "axes": [], "regions": []},
        "candidate_symbols": {
            "points": ["candidate.points.part_000_surface_center"],
            "axes": [],
            "surfaces": ["candidate.surfaces.part_000_surface_center"],
            "parts": [],
            "marks": [],
            "regions": [],
        },
        "role_index": {},
    }


def _relational_cards():
    return {
        "tool_cards": [
            {
                "part_id": "part_000",
                "functional_refs": [
                    {
                        "functional_part_id": "part_000_contact",
                        "executable_tool_refs": {
                            "center_ref": "tool.functional_parts.part_000_contact.center",
                            "normal_ref": "tool.functional_parts.part_000_contact.normal",
                        },
                    }
                ],
                "visual_anchor_evidence": [
                    {
                        "number": 0,
                        "candidate_point_ref": "candidate.points.part_000_surface_center",
                    }
                ],
            }
        ],
        "target_cards": [],
        "target_axis_cards": [{"axis_ref": "scene.axes.motion_axis"}],
    }


def _visual_summary():
    return {
        "numbered_anchor_refs": {
            "rows": [
                {
                    "number": 0,
                    "part_ref": "tool.parts.part_000",
                    "point_ref": "candidate.points.part_000_surface_center",
                    "surface_ref": "candidate.surfaces.part_000_surface_center",
                    "score": 0.8,
                }
            ]
        }
    }


def test_affordance_binding_parse_filter_and_gate() -> None:
    prompt_symbol_table = _prompt_symbol_table()
    allowed_refs = collect_binding_allowed_refs(
        prompt_symbol_table,
        visual_evidence_summary=_visual_summary(),
        relational_keypoint_cards=_relational_cards(),
    )
    assert (
        "tool.functional_parts.part_000_contact.center"
        in allowed_refs["slot_refs"]["function_keypoint_ref"]
    )
    assert "candidate.points.part_000_surface_center" in allowed_refs["visual_anchor_refs"]

    prompt = render_affordance_binding_prompt(
        task_instruction="touch the target with the useful end of the tool",
        symbol_table=prompt_symbol_table,
        visual_evidence=_visual_summary(),
        relational_keypoint_cards=_relational_cards(),
        scene_planning_cards={"target_objects": []},
    )
    assert "visual stability" in prompt

    raw = json.dumps(
        {
            "schema_version": "v0.1",
            "slot_bindings": {
                "grasp_ref": "tool.grasp_references.grasp_center",
                "function_keypoint_ref": "tool.functional_parts.part_000_contact.center",
                "function_axis_ref": "tool.functional_parts.part_000_contact.normal",
                "target_pre_contact_ref": "scene.keypoints.target_pre",
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": "scene.axes.motion_axis",
            }
        }
    )
    binding = parse_affordance_binding_response(
        raw,
        allowed_refs=allowed_refs,
    )

    resolver = ReferenceResolver(
        observed_scene_graph={
            "scene_id": "scene_smoke",
            "objects": {},
            "keypoints": {
                "target_contact": {"xyz_world": [0.1, 0.0, 0.0]},
                "target_pre": {"xyz_world": [0.1, 0.0, 0.1]},
            },
            "axes": {"motion_axis": {"vector_world": [0.0, 0.0, -1.0]}},
            "regions": {},
            "surfaces": {},
        },
        functional_tool_parts={
            "tool_id": "tool_smoke",
            "functional_parts": {
                "part_000_contact": {
                    "center": {"xyz_tool": [0.0, 0.0, 0.0]},
                    "normal": {"vector_tool": [0.0, 0.0, 1.0]},
                    "confidence": 0.9,
                    "source": "unit_test",
                    "provenance": {"observation_id": "obs", "method": "unit"},
                }
            },
            "grasp_references": {
                "grasp_center": {
                    "position_tool": [0.0, 0.0, 0.1],
                    "confidence": 0.9,
                    "source": "unit_test",
                    "provenance": {"observation_id": "obs", "method": "unit"},
                }
            },
        },
    )
    report = check_affordance_binding_sanity(
        binding,
        resolver=resolver,
    )
    assert report["passed"] is True


def test_v2_binding_derives_axis_source_from_selected_candidate_point() -> None:
    prompt_symbol_table = _prompt_symbol_table()
    allowed_refs = collect_binding_allowed_refs(
        prompt_symbol_table,
        visual_evidence_summary=_visual_summary(),
        relational_keypoint_cards=_relational_cards(),
    )
    assert allowed_refs["slot_refs"]["secondary_ref"] == []
    assert "tool.parts.part_000" in allowed_refs["slot_refs"]["axis_source_ref"]

    raw = json.dumps(
        {
            "schema_version": "v0.2",
            "slot_bindings": {
                "grasp_ref": "tool.grasp_references.grasp_center",
                "function_keypoint_ref": "candidate.points.part_000_surface_center",
                "function_axis_ref": None,
                "target_pre_contact_ref": "scene.keypoints.target_pre",
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": "scene.axes.motion_axis",
            },
            "functional_frame_binding": {
                "contact_ref": "candidate.points.part_000_surface_center",
                "primary_ref": "candidate.points.part_000_surface_center",
                "axis_source_ref": None,
                "secondary_ref": None,
                "tertiary_ref": None,
                "parent_part_ref": None,
                "construction": "function_point_center_axis_v1",
            },
            "target_binding": {
                "target_pre_contact_ref": "scene.keypoints.target_pre",
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": "scene.axes.motion_axis",
            },
        }
    )
    binding = parse_affordance_binding_response(raw, allowed_refs=allowed_refs)
    frame = binding["functional_frame_binding"]
    assert frame["contact_ref"] == "candidate.points.part_000_surface_center"
    assert frame["axis_source_ref"] == "tool.parts.part_000"
    assert frame["secondary_ref"] is None
    assert frame["construction"] == "function_point_center_axis_v1"

    resolver = ReferenceResolver(
        observed_scene_graph={
            "scene_id": "scene_smoke",
            "objects": {},
            "keypoints": {
                "target_contact": {"xyz_world": [0.1, 0.0, 0.0]},
                "target_pre": {"xyz_world": [0.1, 0.0, 0.1]},
            },
            "axes": {"motion_axis": {"vector_world": [0.0, 0.0, -1.0]}},
            "regions": {},
            "surfaces": {},
        },
        functional_tool_parts={
            "tool_id": "tool_smoke",
            "functional_parts": {
                "part_000_contact": {
                    "center": {"xyz_tool": [0.0, 0.0, 0.0]},
                    "normal": {"vector_tool": [0.0, 0.0, 1.0]},
                    "confidence": 0.9,
                    "source": "unit_test",
                    "provenance": {"observation_id": "obs", "method": "unit"},
                }
            },
            "grasp_references": {
                "grasp_center": {
                    "position_tool": [0.0, 0.0, 0.1],
                    "confidence": 0.9,
                    "source": "unit_test",
                    "provenance": {"observation_id": "obs", "method": "unit"},
                }
            },
        },
        grounding_candidates={
            "points": {
                "part_000_surface_center": {
                    "xyz_tool": [0.0, 0.0, 0.0],
                    "candidate_type": "geometry_surface_center",
                    "part_ref": "tool.parts.part_000",
                    "confidence": 0.9,
                    "source": "sampart3d_tool_prior",
                    "provenance": {
                        "method": "geometry_surface_center",
                        "source_artifact": "anchors.json",
                    },
                }
            }
        },
        tool_part_graph={
            "tool_id": "tool_smoke",
            "parts": {
                "part_000": {
                    "centroid_xyz_tool": [0.1, 0.0, 0.0],
                    "confidence": 0.9,
                    "source": "sampart3d_tool_prior",
                    "provenance": {"method": "sampart3d", "source_artifact": "parts.json"},
                }
            },
        },
    )
    report = check_affordance_binding_sanity(binding, resolver=resolver)
    assert report["passed"] is True


def test_v2_binding_rejects_new_secondary_point_selection() -> None:
    prompt_symbol_table = _prompt_symbol_table()
    allowed_refs = collect_binding_allowed_refs(
        prompt_symbol_table,
        visual_evidence_summary=_visual_summary(),
        relational_keypoint_cards=_relational_cards(),
    )
    raw = json.dumps(
        {
            "schema_version": "v0.2",
            "slot_bindings": {
                "grasp_ref": "tool.grasp_references.grasp_center",
                "function_keypoint_ref": "candidate.points.part_000_surface_center",
                "function_axis_ref": None,
                "target_pre_contact_ref": "scene.keypoints.target_pre",
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": "scene.axes.motion_axis",
            },
            "functional_frame_binding": {
                "contact_ref": "candidate.points.part_000_surface_center",
                "primary_ref": "candidate.points.part_000_surface_center",
                "axis_source_ref": None,
                "secondary_ref": "candidate.points.part_000_surface_center",
                "tertiary_ref": None,
                "parent_part_ref": "tool.parts.part_000",
                "construction": "function_point_center_axis_v1",
            },
            "target_binding": {
                "target_pre_contact_ref": "scene.keypoints.target_pre",
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": "scene.axes.motion_axis",
            },
        }
    )
    try:
        parse_affordance_binding_response(raw, allowed_refs=allowed_refs)
    except AffordanceBindingError as exc:
        assert "secondary_ref ref is not allowed for that slot" in str(exc)
    else:
        raise AssertionError("secondary_ref should be blocked for new v0.2 bindings")


def test_parent_part_ref_allowed_without_visual_rows() -> None:
    prompt_symbol_table = _prompt_symbol_table()
    allowed_refs = collect_binding_allowed_refs(
        prompt_symbol_table,
        visual_evidence_summary=None,
        relational_keypoint_cards=None,
    )
    assert "tool.parts.part_000" in allowed_refs["slot_refs"]["axis_source_ref"]
    assert "tool.parts.part_000" in allowed_refs["slot_refs"]["parent_part_ref"]

    raw = json.dumps(
        {
            "schema_version": "v0.2",
            "slot_bindings": {
                "grasp_ref": "tool.grasp_references.grasp_center",
                "function_keypoint_ref": "tool.functional_parts.part_000_contact.center",
                "function_axis_ref": None,
                "target_pre_contact_ref": "scene.keypoints.target_pre",
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": "scene.axes.motion_axis",
            },
            "functional_frame_binding": {
                "contact_ref": "tool.functional_parts.part_000_contact.center",
                "primary_ref": "tool.functional_parts.part_000_contact.center",
                "axis_source_ref": "tool.parts.part_000",
                "secondary_ref": None,
                "tertiary_ref": None,
                "parent_part_ref": "tool.parts.part_000",
                "construction": "function_point_center_axis_v1",
            },
            "target_binding": {
                "target_pre_contact_ref": "scene.keypoints.target_pre",
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": "scene.axes.motion_axis",
            },
        }
    )
    binding = parse_affordance_binding_response(raw, allowed_refs=allowed_refs)
    assert binding["functional_frame_binding"]["parent_part_ref"] == "tool.parts.part_000"


def test_geometry_candidate_refs_from_relational_cards_are_bindable() -> None:
    prompt_symbol_table = _prompt_symbol_table()
    prompt_symbol_table["candidate_symbols"]["points"].append(
        "candidate.points.part_000_x_positive_contact"
    )
    relational_cards = {
        "tool_cards": [
            {
                "part_id": "part_000",
                "part_ref": "tool.parts.part_000",
                "candidate_evidence_refs": {
                    "points": ["candidate.points.part_000_x_positive_contact"],
                },
            }
        ],
        "target_cards": [],
        "target_axis_cards": [],
    }
    allowed_refs = collect_binding_allowed_refs(
        prompt_symbol_table,
        visual_evidence_summary=None,
        relational_keypoint_cards=relational_cards,
    )
    assert (
        "candidate.points.part_000_x_positive_contact"
        in allowed_refs["slot_refs"]["function_keypoint_ref"]
    )

    raw = json.dumps(
        {
            "schema_version": "v0.2",
            "slot_bindings": {
                "grasp_ref": None,
                "function_keypoint_ref": "candidate.points.part_000_x_positive_contact",
                "function_axis_ref": None,
                "target_pre_contact_ref": None,
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": None,
            },
            "functional_frame_binding": {
                "contact_ref": "candidate.points.part_000_x_positive_contact",
                "primary_ref": "candidate.points.part_000_x_positive_contact",
                "axis_source_ref": None,
                "secondary_ref": None,
                "tertiary_ref": None,
                "parent_part_ref": None,
                "construction": "function_point_center_axis_v1",
            },
            "target_binding": {
                "target_pre_contact_ref": None,
                "target_contact_ref": "scene.keypoints.target_contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": None,
            },
        }
    )
    binding = parse_affordance_binding_response(raw, allowed_refs=allowed_refs)
    assert binding["functional_frame_binding"]["axis_source_ref"] == "tool.parts.part_000"
