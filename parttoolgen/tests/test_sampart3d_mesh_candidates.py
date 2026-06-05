import numpy as np

from parttoolgen.providers.tool_prior.sampart3d_mesh_candidates import (
    build_sampart3d_mesh_tool_prior,
)


def test_sampart3d_prior_exports_surface_contact_directions(tmp_path):
    mesh_path = tmp_path / "box.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 2 0",
                "v 0 2 0",
                "v 0 0 3",
                "v 1 0 3",
                "v 1 2 3",
                "v 0 2 3",
                "f 1 2 3",
                "f 1 3 4",
                "f 5 7 6",
                "f 5 8 7",
                "f 1 5 6",
                "f 1 6 2",
                "f 2 6 7",
                "f 2 7 3",
                "f 3 7 8",
                "f 3 8 4",
                "f 4 8 5",
                "f 4 5 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path = tmp_path / "mesh_1.0.npy"
    np.save(labels_path, np.zeros(12, dtype=np.int32))

    artifacts = build_sampart3d_mesh_tool_prior(
        tool_id="box_tool",
        source_mesh=mesh_path,
        face_labels_path=labels_path,
        observation_id="box_obs",
    )

    tool_center = artifacts["grounding_candidates"]["points"]["tool_center"]
    assert tool_center["candidate_type"] == "whole_tool_center"
    assert tool_center["xyz_tool"] == [0.5, 1.0, 1.5]
    assert tool_center["provenance"]["method"] == "tool_mesh_bbox_center"

    assert artifacts["grounding_candidates"]["axes"] == {}

    surfaces = artifacts["grounding_candidates"]["surfaces"]
    points = artifacts["grounding_candidates"]["points"]
    assert "part_000_z_positive_endpoint" in points
    assert "part_000_z_positive_endpoint" in surfaces
    assert surfaces["part_000_z_positive_endpoint"]["part_ref"] == "tool.parts.part_000"
    assert surfaces["part_000_z_positive_endpoint"]["candidate_type"] == "tool_part_surface"
