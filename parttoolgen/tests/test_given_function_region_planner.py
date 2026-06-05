from parttoolgen.program.given_function_region_planner import (
    GivenFunctionRegionPlannerError,
    build_given_function_region_program_ir,
    build_region_affordance_binding,
    derive_region_contact_ref,
    normalize_compositional_affordance_program,
)


def _grounding() -> dict:
    return {
        "confidence": 1.0,
        "source": "unit_test",
        "provenance": {
            "observation_id": "unit_test",
            "method": "unit_test",
            "source_artifact": "unit_test",
        },
    }


def _remove_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def _scene() -> dict:
    return {
        "schema_version": "v0.1",
        "scene_id": "scene",
        "frames": {"world": {"type": "world"}},
        "objects": {"target": {**_grounding(), "category": "target_object"}},
        "keypoints": {
            "target_contact_point": {
                **_grounding(),
                "frame": "world",
                "xyz_world": [0.5, 0.0, 0.0],
                "object_ref": "target",
            },
            "target_pre_contact_point": {
                **_grounding(),
                "frame": "world",
                "xyz_world": [0.4, 0.0, 0.0],
                "object_ref": "target",
            },
        },
        "axes": {
            "fallback_axis": {
                **_grounding(),
                "frame": "world",
                "vector_world": [1.0, 0.0, 0.0],
                "object_ref": "target",
            }
        },
        "regions": {},
        "surfaces": {},
        "relations": [],
        "metadata": {"created_by": "unit_test", "geometry_mode": "observed"},
    }


def _target_part_graph() -> dict:
    return {
        "schema_version": "v0.1",
        "graph_id": "target",
        "target_id": "target",
        "frame": "world",
        "parts": {
            "target_surface": {
                **_grounding(),
                "part_ref": "target.parts.target_surface",
                "label": "target_surface",
                "surface_patches": {
                    "top": {
                        **_grounding(),
                        "patch_ref": "target.parts.target_surface.top",
                        "center_xyz_world": [0.5, 0.0, 0.0],
                        "normal_world": [1.0, 0.0, 0.0],
                    }
                },
            }
        },
        "axes": {
            "target_action_down_axis": {
                **_grounding(),
                "axis_ref": "target.axes.target_action_down_axis",
                "vector_world": [1.0, 0.0, 0.0],
            },
            "target_swing_axis": {
                **_grounding(),
                "axis_ref": "target.axes.target_swing_axis",
                "vector_world": [0.0, 1.0, 0.0],
            }
        },
        "relations": [],
        "metadata": {"created_by": "unit_test", "geometry_mode": "mixed"},
    }


def _tool() -> dict:
    return {
        "schema_version": "v0.1",
        "tool_id": "tool",
        "tool_frame": "tool",
        "functional_parts": {
            "face_contact": {
                **_grounding(),
                "role": "contact_patch",
                "center": {"xyz_tool": [0.2, 0.0, 0.0]},
                "normal": {"vector_tool": [1.0, 0.0, 0.0]},
                "axis": {"vector_tool": [0.0, 1.0, 0.0]},
                "frame": {
                    "origin": [0.2, 0.0, 0.0],
                    "x_axis": [0.0, 1.0, 0.0],
                    "y_axis": [0.0, 0.0, 1.0],
                    "z_axis": [1.0, 0.0, 0.0],
                },
                "T_tool_functional": [
                    [0.0, 0.0, 1.0, 0.2],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
        "grasp_references": {
            "grasp": {
                **_grounding(),
                "position_tool": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "metadata": {"created_by": "unit_test", "geometry_mode": "prior"},
    }


def _tool_part_graph() -> dict:
    return {
        "schema_version": "v0.1",
        "tool_id": "tool",
        "tool_frame": "tool",
        "parts": {
            "part_001": {
                **_grounding(),
                "label": "selected",
                "role_candidates": ["functional_candidate"],
                "centroid_xyz_tool": [0.0, 0.0, 0.0],
            }
        },
        "relations": [],
        "metadata": {"created_by": "unit_test", "grounding_mode": "mixed"},
    }


def _candidates() -> dict:
    base = {
        "schema_version": "v0.1",
        "grounding_id": "candidates",
        "observation_id": "unit_test",
        "frames": {"tool": {"frame_type": "tool"}},
        "marks": {},
        "points": {},
        "regions": {},
        "axes": {},
        "surfaces": {},
        "parts": {},
        "metadata": {"created_by": "unit_test", "grounding_mode": "mixed"},
    }
    base["points"]["tool_center"] = {
        **_grounding(),
        "label": "tool_center",
        "candidate_type": "tool_center",
        "xyz_tool": [0.0, 0.0, 0.0],
    }
    base["axes"]["part_001_y_axis"] = {
        **_grounding(),
        "label": "part_001_y_axis",
        "candidate_type": "geometry_axis",
        "part_ref": "tool.parts.part_001",
        "origin_xyz_tool": [0.0, 0.0, 0.0],
        "vector_tool": [0.0, 1.0, 0.0],
    }
    for candidate_id, role, support_faces, xyz in [
        ("part_001_center", "", 0, [0.0, 0.0, 0.0]),
        ("part_001_x_negative_endpoint", "endpoint_contact_patch", 50, [0.1, 0.0, 0.0]),
        ("part_001_y_negative_contact", "contact_patch", 12, [0.2, 0.0, 0.0]),
        ("part_001_y_positive_contact", "contact_patch", 120, [0.3, 0.0, 0.0]),
    ]:
        candidate_type = "geometry_anchor" if candidate_id.endswith("_center") else "geometry_surface_center"
        base["points"][candidate_id] = {
            **_grounding(),
            "label": candidate_id,
            "candidate_type": candidate_type,
            "part_ref": "tool.parts.part_001",
            "xyz_tool": xyz,
            "normal_tool": [1.0, 0.0, 0.0],
            "provenance": {
                **_grounding()["provenance"],
                "role": role,
                "support_faces": support_faces,
                "patch_id": _remove_prefix(candidate_id, "part_001_"),
            },
        }
        if candidate_type == "geometry_surface_center":
            base["surfaces"][candidate_id] = {
                **_grounding(),
                "label": candidate_id,
                "candidate_type": "tool_part_surface",
                "part_ref": "tool.parts.part_001",
                "point_xyz_tool": xyz,
                "normal_tool": [1.0, 0.0, 0.0],
            }
    return base


def _region_selection() -> dict:
    return {
        "schema_version": "v0.2",
        "selected_marker": "2",
        "selected_part_ref": "tool.parts.part_001",
        "selected_part_id": "part_001",
        "confidence": 0.98,
        "reason": "unit test selected region",
    }


def _remap_candidate_part(candidates: dict, *, old_part: str, new_part: str) -> dict:
    remapped = {
        **candidates,
        "points": {},
        "surfaces": {},
        "axes": dict(candidates.get("axes", {})),
    }
    for collection in ("points", "surfaces"):
        for key, value in candidates[collection].items():
            if key.startswith(old_part):
                new_key = key.replace(old_part, new_part, 1)
                remapped[collection][new_key] = {
                    **value,
                    "label": str(value.get("label", key)).replace(old_part, new_part, 1),
                    "part_ref": f"tool.parts.{new_part}",
                }
    return remapped


def _region_selection_for_part(part_id: str) -> dict:
    return {
        **_region_selection(),
        "selected_part_ref": f"tool.parts.{part_id}",
        "selected_part_id": part_id,
    }


def _tool_part_graph_for_part(part_id: str) -> dict:
    graph = _tool_part_graph()
    graph["parts"] = {
        part_id: {
            **_grounding(),
            "label": "selected",
            "role_candidates": ["functional_candidate"],
            "centroid_xyz_tool": [0.0, 0.0, 0.0],
        }
    }
    return graph


def test_region_contact_derivation_requires_vlm_selected_candidate() -> None:
    derived = derive_region_contact_ref(
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        selected_contact_ref="candidate.points.part_001_y_positive_contact",
    )

    assert derived["derived_contact_ref"] == "candidate.points.part_001_y_positive_contact"
    assert derived["derivation"]["method"] == "vlm_selected_candidate_ref"


def test_region_contact_derivation_rejects_region_only_fallback() -> None:
    try:
        derive_region_contact_ref(
            region_selection=_region_selection(),
            grounding_candidates=_candidates(),
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "selected_function_contact_ref" in str(exc)
    else:
        raise AssertionError("region-only contact fallback was accepted")


def test_region_affordance_binding_preserves_region_semantics() -> None:
    binding = build_region_affordance_binding(
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
        target_contact_ref="scene.keypoints.target_contact_point",
        motion_axis_ref="target.axes.target_action_down_axis",
        selected_contact_ref="candidate.points.part_001_y_positive_contact",
        selected_axis_ref="candidate.surfaces.part_001_y_positive_contact",
    )

    assert binding["binding_mode"] == "selected_region_to_target"
    assert binding["slot_bindings"]["function_keypoint_ref"] == "candidate.points.part_001_y_positive_contact"
    assert binding["slot_bindings"]["function_axis_ref"] == "candidate.surfaces.part_001_y_positive_contact"
    assert binding["slot_bindings"]["target_obstacle_ref"] == "scene.objects.target"
    assert binding["functional_frame_binding"]["parent_part_ref"] == "tool.parts.part_001"
    assert binding["region_binding"]["selected_function_region_ref"] == "tool.parts.part_001"


def test_region_planner_rejects_region_only_fixed_template_route() -> None:
    try:
        build_given_function_region_program_ir(
            observed_scene_graph=_scene(),
            functional_tool_parts=_tool(),
            region_selection=_region_selection(),
            task_instruction="hit the target",
            program_id="region_program",
            grounding_candidates=_candidates(),
            tool_part_graph=_tool_part_graph(),
            target_part_graph=_target_part_graph(),
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "Part-Centric Affordance Program" in str(exc)
    else:
        raise AssertionError("region-only fixed-template route was accepted")


def test_region_planner_uses_compositional_vlm_function_point_axis_and_ops() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "approach",
                "ops": [
                    {
                        "type": "near",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "target_pre_contact",
                        "tolerance_m": 0.02,
                    },
                    {
                        "type": "align_axis",
                        "tool_axis_ref": "selected_function_axis_ref",
                        "target_axis_ref": "selected_target_axis_ref",
                    },
                ],
            },
            {
                "name": "engage",
                "ops": [
                    {
                        "type": "contact",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "selected_target_contact_ref",
                    },
                    {
                        "type": "rotate_about",
                        "tool_ref": "selected_function_contact_ref",
                        "start_ref": "target_pre_contact",
                        "goal_ref": "selected_target_contact_ref",
                        "axis_ref": "target.axes.target_swing_axis",
                        "angle_rad": 0.7,
                        "swing_lift_m": 0.05,
                    },
                ],
            },
        ],
        "reason": "compose point, axis, contact, and rotating strike relations",
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )
    assert normalized["schema_version"] == "v0.2-compositional"
    assert normalized["selected_function_axis_ref"] == "candidate.surfaces.part_001_y_positive_contact"
    assert normalized["stages"][1]["ops"][1]["type"] == "rotate_about"

    result = build_given_function_region_program_ir(
        observed_scene_graph=_scene(),
        functional_tool_parts=_tool(),
        region_selection=_region_selection(),
        task_instruction="strike the target with a rotating hammer motion",
        program_id="region_compositional_program",
        grounding_candidates=_candidates(),
        tool_part_graph=_tool_part_graph(),
        target_part_graph=_target_part_graph(),
        part_centric_affordance_program=task_program,
    )

    assert result["part_centric_affordance_program"]["schema_version"] == "v0.2-compositional"
    assert result["program_ir_affordance_contract_report"]["passed"] is True
    assert result["affordance_binding"]["slot_bindings"]["function_axis_ref"] == (
        "candidate.surfaces.part_001_y_positive_contact"
    )
    constraints = result["program_ir"]["constraints"]
    assert any(
        constraint["type"] == "align_axis"
        and constraint["refs"]["source_axis"] == "candidate.surfaces.part_001_y_positive_contact"
        and constraint["refs"]["target_axis"] == "target.axes.target_action_down_axis"
        for constraint in constraints
    )
    assert any(
        constraint["type"] == "move_along"
        and constraint["refs"]["source"] == "candidate.points.part_001_y_positive_contact"
        and constraint["refs"]["swing_axis"] == "target.axes.target_swing_axis"
        and constraint["params"]["profile"] == "swing_arc"
        and constraint["params"]["swing_angle_rad"] == 0.7
        for constraint in constraints
    )


def test_region_planner_accepts_same_contact_tangent_axis_refs() -> None:
    tool = _tool()
    tool["functional_parts"]["part_001_y_positive_contact"] = {
        **_grounding(),
        "role": "contact_patch",
        "center": {"xyz_tool": [0.3, 0.0, 0.0]},
        "normal": {"vector_tool": [1.0, 0.0, 0.0]},
        "axis": {"vector_tool": [0.0, 1.0, 0.0]},
        "tangent_long": {"vector_tool": [0.0, 1.0, 0.0]},
        "tangent_short": {"vector_tool": [0.0, 0.0, 1.0]},
        "frame": {
            "origin": [0.3, 0.0, 0.0],
            "x_axis": [0.0, 1.0, 0.0],
            "y_axis": [0.0, 0.0, 1.0],
            "z_axis": [1.0, 0.0, 0.0],
        },
        "T_tool_functional": [
            [0.0, 0.0, 1.0, 0.3],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    task_program = {
        "schema_version": "v0.3-meta-action",
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "tool.functional_parts.part_001_y_positive_contact.tangent_short",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [
            {
                "name": "approach",
                "ops": [
                    {
                        "type": "align_axis",
                        "tool_axis_ref": "selected_function_axis_ref",
                        "target_axis_ref": "selected_target_axis_ref",
                        "polarity": "auto",
                    },
                    {
                        "type": "contact",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "selected_target_contact_ref",
                    },
                ],
            }
        ],
    }
    result = build_given_function_region_program_ir(
        observed_scene_graph=_scene(),
        functional_tool_parts=tool,
        region_selection=_region_selection(),
        task_instruction="sweep with the short tangent direction",
        program_id="tangent_short_axis",
        grounding_candidates=_candidates(),
        tool_part_graph=_tool_part_graph(),
        target_part_graph=_target_part_graph(),
        part_centric_affordance_program=task_program,
    )
    assert result["affordance_binding"]["slot_bindings"]["function_axis_ref"] == (
        "tool.functional_parts.part_001_y_positive_contact.tangent_short"
    )
    assert any(
        constraint["type"] == "align_axis"
        and constraint["refs"]["source_axis"] == "tool.functional_parts.part_001_y_positive_contact.tangent_short"
        and constraint["params"]["polarity"] == "auto"
        for constraint in result["program_ir"]["constraints"]
    )


def test_region_planner_lowers_part_centric_meta_action_to_program_ir() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_negative_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_negative_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "approach",
                "ops": [
                    {
                        "type": "near",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "target_pre_contact",
                    },
                    {
                        "type": "align_axis",
                        "tool_axis_ref": "selected_function_axis_ref",
                        "target_axis_ref": "selected_target_axis_ref",
                    },
                ],
            },
            {
                "name": "engage",
                "interaction": {
                    "type": "impact_contact",
                    "tool_ref": "selected_function_contact_ref",
                    "target_ref": "selected_target_contact_ref",
                },
                "motion": {
                    "type": "swing_to",
                    "tool_ref": "selected_function_contact_ref",
                    "start_ref": "target_pre_contact",
                    "goal_ref": "selected_target_contact_ref",
                    "axis_ref": "target.axes.target_swing_axis",
                    "angle_rad": 0.8,
                    "swing_lift_m": 0.04,
                },
            },
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )
    assert normalized["schema_version"] == "v0.3-meta-action"
    assert normalized["stages"][1]["interaction"] == [{"type": "impact_contact"}]
    assert normalized["stages"][1]["contact_mode"] == ["impact"]
    assert normalized["stages"][1]["motion"] == [{"type": "rotate"}]

    result = build_given_function_region_program_ir(
        observed_scene_graph=_scene(),
        functional_tool_parts=_tool(),
        region_selection=_region_selection(),
        task_instruction="hit the target with an impact swing",
        program_id="region_meta_action_program",
        grounding_candidates=_candidates(),
        tool_part_graph=_tool_part_graph(),
        target_part_graph=_target_part_graph(),
        part_centric_affordance_program=task_program,
    )

    assert result["part_centric_affordance_program"]["schema_version"] == "v0.3-meta-action"
    constraints = result["program_ir"]["constraints"]
    assert any(
        constraint["type"] == "contact"
        and constraint["refs"]["source"] == "candidate.points.part_001_y_negative_contact"
        and constraint["refs"]["target"] == "scene.keypoints.target_contact_point"
        for constraint in constraints
    )
    assert any(
        constraint["type"] == "align_axis"
        and constraint["refs"]["source_axis"] == "candidate.surfaces.part_001_y_negative_contact"
        and constraint["refs"]["target_axis"] == "target.axes.target_action_down_axis"
        for constraint in constraints
    )
    assert any(
        constraint["type"] == "move_along"
        and constraint["refs"]["source"] == "candidate.points.part_001_y_negative_contact"
        and constraint["refs"]["swing_axis"] == "target.axes.target_swing_axis"
        and constraint["params"]["profile"] == "swing_arc"
        and constraint["params"]["swing_angle_rad"] == 0.8
        and constraint["params"]["swing_lift_m"] == 0.04
        for constraint in constraints
    )


def test_clearance_op_accepts_tool_target_ref_aliases() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [
            {
                "name": "approach",
                "ops": [
                    {
                        "type": "clearance",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "selected_target_contact_ref",
                        "min_distance_m": 0.02,
                    },
                    {
                        "type": "align_axis",
                        "tool_axis_ref": "selected_function_axis_ref",
                        "target_axis_ref": "selected_target_axis_ref",
                    },
                    {
                        "type": "contact",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "selected_target_contact_ref",
                    },
                ],
            }
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )

    clearance = normalized["stages"][0]["ops"][0]
    assert clearance["type"] == "clearance"
    assert clearance["moving_ref"] == "candidate.points.part_001_y_positive_contact"
    assert clearance["obstacle_ref"] == "scene.keypoints.target_contact_point"


def test_clearance_op_defaults_moving_ref_and_accepts_object_aliases() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [
            {
                "name": "approach",
                "ops": [
                    {
                        "type": "clearance",
                        "object_ref": "scene.objects.target",
                        "min_clearance_m": 0.003,
                    }
                ],
            }
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )

    clearance = normalized["stages"][0]["ops"][0]
    assert clearance["type"] == "clearance"
    assert clearance["moving_ref"] == "candidate.points.part_001_y_positive_contact"
    assert clearance["obstacle_ref"] == "scene.objects.target"
    assert clearance["min_distance_m"] == 0.003


def test_normalized_meta_action_reparse_does_not_expand_motion_summary_again() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_negative_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_negative_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "engage",
                "interaction": {
                    "type": "impact_contact",
                    "tool_ref": "selected_function_contact_ref",
                    "target_ref": "selected_target_contact_ref",
                },
                "motion": {
                    "type": "swing_to",
                    "tool_ref": "selected_function_contact_ref",
                    "start_ref": "target_pre_contact",
                    "goal_ref": "selected_target_contact_ref",
                    "axis_ref": "target.axes.target_swing_axis",
                    "angle_rad": 0.8,
                    "swing_lift_m": 0.04,
                },
            },
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )
    reparsed = normalize_compositional_affordance_program(
        part_centric_affordance_program=normalized,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )

    rotate_ops = [
        op
        for stage in reparsed["stages"]
        for op in stage["ops"]
        if op["type"] == "rotate_about"
    ]
    assert len(rotate_ops) == 1
    assert rotate_ops[0]["swing_axis_ref"] == "target.axes.target_swing_axis"


def test_skeletal_motion_summary_with_explicit_ops_does_not_default_swing_axis() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_negative_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_negative_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "engage",
                "interaction": {"type": "impact_contact"},
                "motion": {"type": "swing_to"},
                "ops": [
                    {
                        "type": "rotate_about",
                        "tool_ref": "selected_function_contact_ref",
                        "start_ref": "target_pre_contact",
                        "goal_ref": "selected_target_contact_ref",
                        "axis_ref": "target.axes.target_swing_axis",
                        "angle_rad": 0.8,
                        "swing_lift_m": 0.04,
                    }
                ],
            },
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )

    rotate_ops = [
        op
        for stage in normalized["stages"]
        for op in stage["ops"]
        if op["type"] == "rotate_about"
    ]
    assert len(rotate_ops) == 1
    assert rotate_ops[0]["swing_axis_ref"] == "target.axes.target_swing_axis"


def test_summary_only_motion_without_executable_motion_op_is_dropped() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_negative_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_negative_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "engage",
                "interaction": {"type": "impact_contact"},
                "motion": {"type": "swing_to"},
                "ops": [
                    {
                        "type": "align_axis",
                        "tool_axis_ref": "selected_function_axis_ref",
                        "target_axis_ref": "selected_target_axis_ref",
                    },
                    {
                        "type": "contact",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "selected_target_contact_ref",
                    },
                ],
            },
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )

    assert normalized["stages"][0]["motion"] == []
    assert [op["type"] for op in normalized["stages"][0]["ops"]] == [
        "align_axis",
        "contact",
    ]


def test_meta_action_translate_and_rotate_motion_aliases_lower_to_existing_ops() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "translate",
                "interaction": {"type": "surface_contact"},
                "motion": {
                    "type": "translate",
                    "tool_ref": "selected_function_contact_ref",
                    "start_ref": "target_pre_contact",
                    "goal_ref": "selected_target_contact_ref",
                },
            },
            {
                "name": "rotate",
                "interaction": {"type": "impact_contact"},
                "motion": {
                    "type": "rotate",
                    "tool_ref": "selected_function_contact_ref",
                    "start_ref": "target_pre_contact",
                    "goal_ref": "selected_target_contact_ref",
                    "axis_ref": "target.axes.target_swing_axis",
                    "angle_rad": 0.7,
                    "swing_lift_m": 0.02,
                },
            },
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )

    translate_op = normalized["stages"][0]["ops"][-1]
    assert normalized["stages"][0]["motion"] == [{"type": "translate"}]
    assert translate_op["type"] == "move_along"
    assert translate_op["profile"] == "linear"
    assert translate_op["path_family"] == "contact_follow"

    rotate_op = normalized["stages"][1]["ops"][-1]
    assert normalized["stages"][1]["motion"] == [{"type": "rotate"}]
    assert rotate_op["type"] == "rotate_about"
    assert rotate_op["profile"] == "swing_arc"
    assert rotate_op["path_family"] == "swing"
    assert rotate_op["swing_axis_ref"] == "target.axes.target_swing_axis"
    assert rotate_op["swing_angle_rad"] == 0.7
    assert rotate_op["swing_lift_m"] == 0.02


def test_contact_mode_translate_program_lowers_to_surface_contact_ops() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "engage",
                "contact_mode": {"type": "sustained_contact"},
                "motion": {
                    "type": "translate",
                    "tool_ref": "selected_function_contact_ref",
                    "start_ref": "target_pre_contact",
                    "goal_ref": "selected_target_contact_ref",
                },
            }
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )

    assert normalized["stages"][0]["contact_mode"] == ["sustained_contact"]
    assert normalized["stages"][0]["interaction"] == []
    assert normalized["stages"][0]["motion"] == [{"type": "translate"}]
    assert [op["type"] for op in normalized["stages"][0]["ops"]] == [
        "align_axis",
        "contact",
        "move_along",
    ]
    assert normalized["stages"][0]["ops"][1]["tool_ref"] == "candidate.points.part_001_y_positive_contact"
    assert normalized["stages"][0]["ops"][1]["target_ref"] == "scene.keypoints.target_contact_point"


def test_contact_mode_impact_rotate_program_lowers_to_swing_arc_ops() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_negative_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_negative_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "engage",
                "contact_mode": "impact",
                "motion": {
                    "type": "rotate",
                    "tool_ref": "selected_function_contact_ref",
                    "start_ref": "target_pre_contact",
                    "goal_ref": "selected_target_contact_ref",
                    "axis_ref": "target.axes.target_swing_axis",
                    "angle_rad": 0.6,
                    "swing_lift_m": 0.025,
                },
            }
        ],
    }

    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=task_program,
        region_selection=_region_selection(),
        grounding_candidates=_candidates(),
        observed_scene_graph=_scene(),
        target_part_graph=_target_part_graph(),
    )

    rotate_op = normalized["stages"][0]["ops"][-1]
    assert normalized["stages"][0]["contact_mode"] == ["impact"]
    assert normalized["stages"][0]["motion"] == [{"type": "rotate"}]
    assert rotate_op["type"] == "rotate_about"
    assert rotate_op["profile"] == "swing_arc"
    assert rotate_op["path_family"] == "swing"
    assert rotate_op["swing_axis_ref"] == "target.axes.target_swing_axis"
    assert rotate_op["swing_angle_rad"] == 0.6
    assert rotate_op["swing_lift_m"] == 0.025


def test_meta_action_surface_slide_and_press_lower_to_generic_constraints() -> None:
    slide_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
        "stages": [
            {
                "name": "slide",
                "interaction": {"type": "surface_contact"},
                "motion": {
                    "type": "slide_along",
                    "tool_ref": "selected_function_contact_ref",
                    "start_ref": "target_pre_contact",
                    "goal_ref": "selected_target_contact_ref",
                },
            }
        ],
    }
    slide = build_given_function_region_program_ir(
        observed_scene_graph=_scene(),
        functional_tool_parts=_tool(),
        region_selection=_region_selection(),
        task_instruction="slide along the target surface",
        program_id="surface_slide_meta_action",
        grounding_candidates=_candidates(),
        tool_part_graph=_tool_part_graph(),
        target_part_graph=_target_part_graph(),
        part_centric_affordance_program=slide_program,
    )
    slide_constraints = slide["program_ir"]["constraints"]
    assert any(constraint["type"] == "align_axis" for constraint in slide_constraints)
    assert any(constraint["type"] == "contact" for constraint in slide_constraints)
    assert any(
        constraint["type"] == "move_along"
        and constraint["refs"]["start"] == "scene.keypoints.target_pre_contact_point"
        and constraint["refs"]["end"] == "scene.keypoints.target_contact_point"
        for constraint in slide_constraints
    )

    press_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [
            {
                "name": "press",
                "interaction": {"type": "point_contact"},
                "motion": {"type": "press_along", "distance_m": 0.04},
            }
        ],
    }
    press = build_given_function_region_program_ir(
        observed_scene_graph=_scene(),
        functional_tool_parts=_tool(),
        region_selection=_region_selection(),
        task_instruction="press the target",
        program_id="press_meta_action",
        grounding_candidates=_candidates(),
        tool_part_graph=_tool_part_graph(),
        target_part_graph=_target_part_graph(),
        part_centric_affordance_program=press_program,
    )
    press_constraints = press["program_ir"]["constraints"]
    assert any(
        constraint["type"] == "pre_contact_offset"
        and constraint["params"]["distance_m"] == 0.04
        for constraint in press_constraints
    )
    assert any(constraint["type"] == "contact" for constraint in press_constraints)


def test_meta_action_repeated_primitive_invocations_keep_distinct_targets() -> None:
    scene = _scene()
    scene["keypoints"]["target_contact_point_2"] = {
        **_grounding(),
        "frame": "world",
        "xyz_world": [0.6, 0.0, 0.0],
        "object_ref": "target",
    }
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [
            {
                "name": "press_first",
                "interaction": {"type": "point_contact", "target_ref": "scene.keypoints.target_contact_point"},
                "motion": {"type": "press_along", "target_ref": "scene.keypoints.target_contact_point"},
            },
            {
                "name": "press_second",
                "interaction": {"type": "point_contact", "target_ref": "scene.keypoints.target_contact_point_2"},
                "motion": {"type": "press_along", "target_ref": "scene.keypoints.target_contact_point_2"},
            },
        ],
    }

    result = build_given_function_region_program_ir(
        observed_scene_graph=scene,
        functional_tool_parts=_tool(),
        region_selection=_region_selection(),
        task_instruction="press two target points with the same tool primitive",
        program_id="repeated_press_meta_action",
        grounding_candidates=_candidates(),
        tool_part_graph=_tool_part_graph(),
        target_part_graph=_target_part_graph(),
        part_centric_affordance_program=task_program,
    )
    contacts = [
        constraint
        for constraint in result["program_ir"]["constraints"]
        if constraint["type"] == "contact"
    ]
    contacted_targets = {constraint["refs"]["target"] for constraint in contacts}
    assert "scene.keypoints.target_contact_point" in contacted_targets
    assert "scene.keypoints.target_contact_point_2" in contacted_targets
    assert [stage["name"] for stage in result["program_ir"]["stages"]] == [
        "press_first",
        "press_second",
    ]


def test_meta_action_primitive_reuse_is_ref_driven_not_part_name_driven() -> None:
    def build_for_part(part_id: str):
        candidates = _candidates() if part_id == "part_001" else _remap_candidate_part(
            _candidates(),
            old_part="part_001",
            new_part=part_id,
        )
        task_program = {
            "selected_tool_region_ref": f"tool.parts.{part_id}",
            "selected_function_contact_ref": f"candidate.points.{part_id}_y_positive_contact",
            "selected_function_axis_ref": f"candidate.surfaces.{part_id}_y_positive_contact",
            "selected_target_contact_ref": "scene.keypoints.target_contact_point",
            "selected_target_axis_ref": "target.axes.target_action_down_axis",
            "target_pre_contact_ref": "scene.keypoints.target_pre_contact_point",
            "stages": [
                {
                    "name": "surface_slide",
                    "interaction": {"type": "surface_contact"},
                    "motion": {
                        "type": "slide_along",
                        "start_ref": "target_pre_contact",
                        "goal_ref": "selected_target_contact_ref",
                    },
                }
            ],
        }
        result = build_given_function_region_program_ir(
            observed_scene_graph=_scene(),
            functional_tool_parts=_tool(),
            region_selection=_region_selection_for_part(part_id),
            task_instruction="reuse surface slide primitive on another part",
            program_id=f"primitive_reuse_{part_id}",
            grounding_candidates=candidates,
            tool_part_graph=_tool_part_graph_for_part(part_id),
            target_part_graph=_target_part_graph(),
            part_centric_affordance_program=task_program,
        )
        return [
            (constraint["type"], tuple(sorted(constraint["refs"])))
            for constraint in result["program_ir"]["constraints"]
        ]

    assert build_for_part("part_001") == build_for_part("part_002")


def test_meta_action_program_rejects_unknown_interaction_primitive() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [
            {
                "name": "engage",
                "interaction": {"type": "hammer_only_magic"},
            }
        ],
    }
    try:
        normalize_compositional_affordance_program(
            part_centric_affordance_program=task_program,
            region_selection=_region_selection(),
            grounding_candidates=_candidates(),
            observed_scene_graph=_scene(),
            target_part_graph=_target_part_graph(),
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "interaction primitive" in str(exc)
    else:
        raise AssertionError("unknown interaction primitive was accepted")


def test_part_centric_program_rejects_tool_category_template_metadata() -> None:
    task_program = {
        "tool_category": "hammer",
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_positive_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [
            {
                "name": "engage",
                "stage_template": "hammer_strike",
                "ops": [
                    {
                        "type": "contact",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "selected_target_contact_ref",
                    }
                ],
            }
        ],
    }
    try:
        normalize_compositional_affordance_program(
            part_centric_affordance_program=task_program,
            region_selection=_region_selection(),
            grounding_candidates=_candidates(),
            observed_scene_graph=_scene(),
            target_part_graph=_target_part_graph(),
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "category/template metadata" in str(exc)
    else:
        raise AssertionError("tool-category template metadata was accepted")


def test_compositional_program_rejects_invented_function_axis_ref() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.fake_axis",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [{"name": "engage", "ops": [{"type": "contact", "tool_ref": "selected_function_contact_ref", "target_ref": "selected_target_contact_ref"}]}],
    }
    try:
        normalize_compositional_affordance_program(
            part_centric_affordance_program=task_program,
            region_selection=_region_selection(),
            grounding_candidates=_candidates(),
            observed_scene_graph=_scene(),
            target_part_graph=_target_part_graph(),
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "selected_function_axis_ref" in str(exc)
    else:
        raise AssertionError("invented function axis ref was accepted")


def test_compositional_program_rejects_mismatched_contact_surface_axis() -> None:
    task_program = {
        "selected_tool_region_ref": "tool.parts.part_001",
        "selected_function_contact_ref": "candidate.points.part_001_y_positive_contact",
        "selected_function_axis_ref": "candidate.surfaces.part_001_y_negative_contact",
        "selected_target_contact_ref": "scene.keypoints.target_contact_point",
        "selected_target_axis_ref": "target.axes.target_action_down_axis",
        "stages": [
            {
                "name": "engage",
                "ops": [
                    {
                        "type": "contact",
                        "tool_ref": "selected_function_contact_ref",
                        "target_ref": "selected_target_contact_ref",
                    }
                ],
            }
        ],
    }
    try:
        normalize_compositional_affordance_program(
            part_centric_affordance_program=task_program,
            region_selection=_region_selection(),
            grounding_candidates=_candidates(),
            observed_scene_graph=_scene(),
            target_part_graph=_target_part_graph(),
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "matching selected_function_contact_ref" in str(exc)
    else:
        raise AssertionError("mismatched function contact/surface refs were accepted")


def test_region_binding_rejects_missing_explicit_target_ref() -> None:
    try:
        build_region_affordance_binding(
            region_selection=_region_selection(),
            grounding_candidates=_candidates(),
            observed_scene_graph=_scene(),
            target_contact_ref="scene.keypoints.missing_target",
            motion_axis_ref="scene.axes.fallback_axis",
            selected_contact_ref="candidate.points.part_001_y_positive_contact",
            selected_axis_ref="candidate.surfaces.part_001_y_positive_contact",
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "target_contact_ref is required" in str(exc)
    else:
        raise AssertionError("missing target contact ref was silently replaced")


def test_region_binding_requires_derived_surface_axis() -> None:
    candidates = _candidates()
    candidates["surfaces"] = {}

    try:
        build_region_affordance_binding(
            region_selection=_region_selection(),
            grounding_candidates=candidates,
            observed_scene_graph=_scene(),
            target_part_graph=_target_part_graph(),
            target_contact_ref="scene.keypoints.target_contact_point",
            motion_axis_ref="target.axes.target_action_down_axis",
            selected_contact_ref="candidate.points.part_001_y_positive_contact",
            selected_axis_ref="candidate.surfaces.part_001_y_positive_contact",
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "function_axis_ref" in str(exc)
    else:
        raise AssertionError("selected-region mainline accepted a contact point without surface axis")


def test_region_binding_requires_motion_axis() -> None:
    try:
        build_region_affordance_binding(
            region_selection=_region_selection(),
            grounding_candidates=_candidates(),
            observed_scene_graph=_scene(),
            target_part_graph=_target_part_graph(),
            target_contact_ref="scene.keypoints.target_contact_point",
            selected_contact_ref="candidate.points.part_001_y_positive_contact",
            selected_axis_ref="candidate.surfaces.part_001_y_positive_contact",
            motion_axis_ref=None,
        )
    except GivenFunctionRegionPlannerError as exc:
        assert "motion_axis_ref" in str(exc)
    else:
        raise AssertionError("selected-region mainline accepted a missing motion axis")
