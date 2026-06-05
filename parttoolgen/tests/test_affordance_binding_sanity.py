from typing import Dict

from parttoolgen.checks.affordance_binding_sanity import (
    check_affordance_binding_sanity,
    check_program_ir_affordance_contract,
    check_program_ir_affordance_bindings,
)
from parttoolgen.compiler.reference_resolver import ReferenceResolver
from parttoolgen.program.repair import summarize_compiler_report


def _resolver() -> ReferenceResolver:
    scene = {
        "scene_id": "scene",
        "metadata": {"geometry_mode": "observed"},
        "objects": {},
        "keypoints": {
            "contact": {
                "frame": "world",
                "xyz_world": [1.0, 0.0, 0.0],
                "object_ref": "scene.objects.target",
                "confidence": 0.9,
                "source": "rgbd",
                "provenance": {
                    "method": "unit_test",
                    "source_artifact": "scene.json",
                },
            },
            "secondary": {
                "frame": "world",
                "xyz_world": [1.1, 0.0, 0.0],
                "object_ref": "scene.objects.target",
                "confidence": 0.9,
                "source": "rgbd",
                "provenance": {
                    "method": "unit_test",
                    "source_artifact": "scene.json",
                },
            },
        },
        "axes": {
            "approach": {
                "frame": "world",
                "vector_world": [0.0, 0.0, 1.0],
                "object_ref": "scene.objects.target",
                "confidence": 0.9,
                "source": "rgbd",
                "provenance": {
                    "method": "unit_test",
                    "source_artifact": "scene.json",
                },
            }
        },
        "regions": {},
        "surfaces": {},
    }
    tool = {
        "tool_id": "tool",
        "functional_parts": {
            "face": {
                "center": {"xyz_tool": [0.0, 0.0, 0.0]},
                "normal": {"vector_tool": [0.0, 0.0, 1.0]},
                "axis": {"vector_tool": [1.0, 0.0, 0.0]},
                "confidence": 0.95,
                "source": "mesh_prior",
                "provenance": {
                    "method": "unit_test",
                    "source_artifact": "tool.json",
                },
            }
        },
        "grasp_references": {},
    }
    candidates = {
        "points": {
            "surface_center": {
                "xyz_tool": [0.0, 0.0, 0.0],
                "candidate_type": "geometry_surface_center",
                "part_ref": "tool.parts.face",
                "confidence": 0.99,
                "source": "sampart3d_tool_prior",
                "patch": [3, 5],
                "provenance": {
                    "method": "geometry_surface_center",
                    "anchor_set_id": "anchors",
                    "source_artifact": "anchors.json",
                },
            },
            "world_frame_candidate": {
                "xyz_world": [1.0, 0.1, 0.0],
                "candidate_type": "world_frame_candidate",
                "confidence": 0.8,
                "source": "unit_test_world_frame",
                "provenance": {
                    "method": "world_frame_fixture",
                    "source_artifact": "world_frame_fixture.json",
                },
            }
        },
        "marks": {},
        "regions": {},
        "axes": {},
        "surfaces": {},
        "parts": {},
    }
    return ReferenceResolver(
        observed_scene_graph=scene,
        functional_tool_parts=tool,
        grounding_candidates=candidates,
    )


def test_affordance_binding_passes_for_functional_part_and_evidence() -> None:
    report = check_affordance_binding_sanity(
        {
            "binding_id": "binding",
            "required_slots": [
                "tool_contact_point",
                "tool_contact_axis",
                "target_contact",
                "target_approach_axis",
            ],
            "slots": {
                "tool_contact_point": "tool.functional_parts.face.center",
                "tool_contact_axis": "tool.functional_parts.face.normal",
                "target_contact": "scene.keypoints.contact",
                "target_approach_axis": "scene.axes.approach",
            },
            "secondary_refs": ["scene.keypoints.secondary"],
            "evidence_refs": ["candidate.points.surface_center"],
        },
        resolver=_resolver(),
    )

    assert report["passed"] is True
    assert report["failed_refs"] == []
    assert any(check["name"] == "point_axis_same_part_or_provenance" for check in report["checks"])


def test_vlm_affordance_slot_bindings_shape_is_supported() -> None:
    report = check_affordance_binding_sanity(
        {
            "schema_version": "v0.1",
            "slot_bindings": {
                "grasp_ref": None,
                "function_keypoint_ref": "tool.functional_parts.face.center",
                "function_axis_ref": "tool.functional_parts.face.normal",
                "target_pre_contact_ref": "scene.keypoints.secondary",
                "target_contact_ref": "scene.keypoints.contact",
                "target_post_contact_ref": None,
                "motion_axis_ref": "scene.axes.approach",
            },
            "evidence": {"refs": ["candidate.points.surface_center"]},
        },
        resolver=_resolver(),
    )

    assert report["passed"] is True
    assert report["failed_refs"] == []


def test_candidate_anchor_is_not_executable_tool_contact_by_default() -> None:
    report = check_affordance_binding_sanity(
        {
            "binding_id": "bad_contact",
            "type": "contact",
            "refs": {
                "source": "candidate.points.world_frame_candidate",
                "target": "scene.keypoints.contact",
            },
        },
        resolver=_resolver(),
    )

    assert report["passed"] is False
    assert "candidate.points.world_frame_candidate" in report["failed_refs"]
    assert any(check["name"] == "candidate_not_physical_contact" for check in report["checks"])


def test_explicitly_promoted_candidate_can_be_used_as_tool_contact() -> None:
    report = check_affordance_binding_sanity(
        {
            "binding_id": "promoted_contact",
            "type": "contact",
            "refs": {
                "source": {"ref": "candidate.points.surface_center", "promoted": True},
                "target": "scene.keypoints.contact",
            },
        },
        resolver=_resolver(),
    )

    assert report["passed"] is True
    assert "candidate.points.surface_center" not in report["failed_refs"]


def test_axis_slot_rejects_zero_vector_from_lookup() -> None:
    def lookup(ref: str) -> Dict[str, object]:
        assert ref == "tool.functional_parts.face.axis"
        return {
            "kind": "axis",
            "frame": "tool",
            "entity": {
                "vector_tool": [0.0, 0.0, 0.0],
                "confidence": 0.9,
                "provenance": {"method": "unit_test"},
            },
            "vector_tool": [0.0, 0.0, 0.0],
        }

    report = check_affordance_binding_sanity(
        {
            "binding_id": "zero_axis",
            "required_slots": ["tool_contact_axis"],
            "slots": {"tool_contact_axis": "tool.functional_parts.face.axis"},
        },
        symbol_lookup=lookup,
    )

    assert report["passed"] is False
    assert "tool.functional_parts.face.axis" in report["failed_refs"]
    assert any(check["name"] == "slot_geometry_type" for check in report["checks"])


def test_distance_binding_requires_tool_and_world_points() -> None:
    report = check_affordance_binding_sanity(
        {
            "binding_id": "bad_distance",
            "type": "distance",
            "refs": {
                "a": "scene.keypoints.contact",
                "b": "scene.keypoints.secondary",
            },
        },
        resolver=_resolver(),
    )

    assert report["passed"] is False
    assert any(check["name"] == "constraint_frame_consistency" for check in report["checks"])


def test_program_ir_affordance_binding_allows_tool_prior_candidate_source() -> None:
    report = check_program_ir_affordance_bindings(
        {
            "program_id": "program",
            "constraints": [
                {
                    "id": "contact_1",
                    "type": "contact",
                    "refs": {
                        "source": "candidate.points.surface_center",
                        "target": "scene.keypoints.contact",
                    },
                }
            ],
        },
        resolver=_resolver(),
    )

    assert report["passed"] is True
    assert report["program_id"] == "program"
    assert report["failed_refs"] == []


def test_program_ir_affordance_contract_requires_derived_alignment() -> None:
    binding = {
        "binding_id": "binding",
        "slot_bindings": {
            "function_keypoint_ref": "tool.functional_parts.face.center",
            "target_pre_contact_ref": "scene.keypoints.secondary",
            "target_contact_ref": "scene.keypoints.contact",
            "target_post_contact_ref": "scene.keypoints.contact",
            "motion_axis_ref": "scene.axes.approach",
        },
        "functional_frame_binding": {
            "contact_ref": "tool.functional_parts.face.center",
            "axis_source_ref": "tool.parts.face",
        },
    }
    report = check_program_ir_affordance_contract(
        {
            "program_id": "program",
            "constraints": [
                {
                    "id": "approach",
                    "type": "contact",
                    "refs": {
                        "source": "tool.functional_parts.face.center",
                        "target": "scene.keypoints.secondary",
                    },
                },
                {
                    "id": "engage",
                    "type": "contact",
                    "refs": {
                        "source": "tool.functional_parts.face.center",
                        "target": "scene.keypoints.contact",
                    },
                },
                {
                    "id": "move",
                    "type": "move_along",
                        "refs": {
                            "source": "tool.functional_parts.face.center",
                            "start": "scene.keypoints.secondary",
                            "end": "scene.keypoints.contact",
                        },
                    },
            ],
        },
        binding,
    )

    assert report["passed"] is False
    assert any(check["name"] == "derived_function_axis_aligned" for check in report["checks"])


def test_program_ir_affordance_contract_requires_explicit_axis_alignment() -> None:
    binding = {
        "binding_id": "binding",
        "slot_bindings": {
            "function_keypoint_ref": "tool.functional_parts.face.center",
            "function_axis_ref": "tool.functional_parts.face.normal",
            "target_contact_ref": "scene.keypoints.contact",
            "motion_axis_ref": "scene.axes.approach",
        },
        "functional_frame_binding": {
            "contact_ref": "tool.functional_parts.face.center",
            "axis_source_ref": "tool.parts.face",
        },
    }
    report = check_program_ir_affordance_contract(
        {
            "program_id": "program",
            "constraints": [
                {
                    "id": "contact",
                    "type": "contact",
                    "refs": {
                        "source": "tool.functional_parts.face.center",
                        "target": "scene.keypoints.contact",
                    },
                }
            ],
        },
        binding,
    )

    assert report["passed"] is False
    assert any(check["name"] == "explicit_function_axis_aligned" for check in report["checks"])


def test_selected_region_contract_requires_explicit_axis_binding() -> None:
    binding = {
        "binding_id": "binding",
        "binding_mode": "selected_region_to_target",
        "slot_bindings": {
            "function_keypoint_ref": "tool.functional_parts.face.center",
            "target_contact_ref": "scene.keypoints.contact",
            "motion_axis_ref": "scene.axes.approach",
        },
        "functional_frame_binding": {
            "contact_ref": "tool.functional_parts.face.center",
            "axis_source_ref": "tool.parts.face",
            "construction": "selected_region_contact_patch_v1",
        },
    }
    report = check_program_ir_affordance_contract(
        {
            "program_id": "program",
            "constraints": [
                {
                    "id": "point_pair_axis",
                    "type": "align_vector_between_points",
                    "refs": {
                        "source_start": "tool.parts.face",
                        "source_end": "tool.functional_parts.face.center",
                        "target_axis": "scene.axes.approach",
                    },
                },
                {
                    "id": "contact",
                    "type": "contact",
                    "refs": {
                        "source": "tool.functional_parts.face.center",
                        "target": "scene.keypoints.contact",
                    },
                },
            ],
        },
        binding,
    )

    assert report["passed"] is False
    assert any(check["name"] == "explicit_function_axis_bound" for check in report["checks"])
    assert not any(check["name"] == "derived_function_axis_aligned" for check in report["checks"])


def test_program_ir_affordance_contract_passes_explicit_axis_alignment() -> None:
    binding = {
        "binding_id": "binding",
        "slot_bindings": {
            "function_keypoint_ref": "tool.functional_parts.face.center",
            "function_axis_ref": "tool.functional_parts.face.normal",
            "target_contact_ref": "scene.keypoints.contact",
            "motion_axis_ref": "scene.axes.approach",
        },
        "functional_frame_binding": {
            "contact_ref": "tool.functional_parts.face.center",
            "axis_source_ref": "tool.parts.face",
        },
    }
    report = check_program_ir_affordance_contract(
        {
            "program_id": "program",
            "constraints": [
                {
                    "id": "align",
                    "type": "align_axis",
                    "refs": {
                        "source_axis": "tool.functional_parts.face.normal",
                        "target_axis": "scene.axes.approach",
                    },
                },
                {
                    "id": "contact",
                    "type": "contact",
                    "refs": {
                        "source": "tool.functional_parts.face.center",
                        "target": "scene.keypoints.contact",
                    },
                },
            ],
        },
        binding,
    )

    assert report["passed"] is True


def test_program_ir_affordance_contract_passes_clean_given_point_program() -> None:
    binding = {
        "binding_id": "binding",
        "slot_bindings": {
            "function_keypoint_ref": "tool.functional_parts.face.center",
            "target_pre_contact_ref": "scene.keypoints.secondary",
            "target_contact_ref": "scene.keypoints.contact",
            "target_post_contact_ref": "scene.keypoints.contact",
            "motion_axis_ref": "scene.axes.approach",
        },
        "functional_frame_binding": {
            "contact_ref": "tool.functional_parts.face.center",
            "axis_source_ref": "tool.parts.face",
        },
    }
    report = check_program_ir_affordance_contract(
        {
            "program_id": "program",
            "constraints": [
                {
                    "id": "align",
                    "type": "align_vector_between_points",
                    "refs": {
                        "source_start": "tool.parts.face",
                        "source_end": "tool.functional_parts.face.center",
                        "target_axis": "scene.axes.approach",
                    },
                },
                {
                    "id": "contact",
                    "type": "contact",
                    "refs": {
                        "source": "tool.functional_parts.face.center",
                        "target": "scene.keypoints.contact",
                    },
                },
                {
                    "id": "move",
                    "type": "move_along",
                        "refs": {
                            "source": "tool.functional_parts.face.center",
                            "start": "scene.keypoints.secondary",
                            "end": "scene.keypoints.contact",
                        },
                    },
            ],
        },
        binding,
    )

    assert report["passed"] is True
    assert report["program_id"] == "program"


def test_secondary_refs_must_be_non_degenerate() -> None:
    report = check_affordance_binding_sanity(
        {
            "binding_id": "degenerate_secondary",
            "slots": {"tool_contact_point": "tool.functional_parts.face.center"},
            "secondary_refs": ["tool.functional_parts.face.center"],
        },
        resolver=_resolver(),
    )

    assert report["passed"] is False
    assert any(check["name"] == "secondary_refs_non_degenerate" for check in report["checks"])


def test_repair_summary_includes_affordance_binding_sanity() -> None:
    binding_report = check_affordance_binding_sanity(
        {
            "binding_id": "bad_contact",
            "type": "contact",
            "refs": {
                "source": "candidate.points.world_frame_candidate",
                "target": "scene.keypoints.contact",
            },
        },
        resolver=_resolver(),
    )
    summary = summarize_compiler_report(
        {
            "passed": False,
            "stages": [],
            "metadata": {"affordance_binding_sanity": binding_report},
        }
    )

    assert "candidate.points.world_frame_candidate" in summary["failed_refs"]
    assert summary["affordance_binding_sanity"]["repair_hints"]
