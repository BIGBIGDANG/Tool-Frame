from __future__ import annotations

from parttoolgen.evaluation.marker_trace_ink_evidence import _extract_tip_row
from parttoolgen.evaluation.side_board_hdu_erase_success import _selected_contact_row


def test_side_board_hdu_evaluator_accepts_current_contact_selection_schema() -> None:
    row = {
        "point_ref": "candidate.points.part_292_y_negative_contact",
        "xyz_tool": [0.03, -0.01, 0.0],
        "normal_tool": [0.0, -1.0, 0.0],
    }
    selection = {
        "contact_point": {
            "marker": "F3",
            "point_ref": row["point_ref"],
            "row": row,
        },
        "contact_point_candidates": [
            {
                "marker": "F3",
                "point_ref": row["point_ref"],
                "row": row,
            }
        ],
    }

    assert _selected_contact_row(selection) == row


def test_marker_ink_evaluator_accepts_current_contact_selection_schema() -> None:
    row = {
        "point_ref": "candidate.points.marker_tip",
        "xyz_tool": [0.0, 0.0, 0.04],
        "normal_tool": [0.0, 0.0, -1.0],
    }
    selection = {
        "contact_point": {
            "marker": "F5",
            "point_ref": row["point_ref"],
            "row": row,
        },
        "contact_point_candidates": [
            {
                "marker": "F5",
                "point_ref": row["point_ref"],
                "row": row,
            }
        ],
    }

    assert _extract_tip_row(selection) == row
