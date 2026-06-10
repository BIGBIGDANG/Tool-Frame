#!/usr/bin/env python3
"""Use a VLM judge to score marker-trace ink recognizability."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from parttoolgen.constraints.vlm_client import build_vlm_client


def _load_json(path: str | Path) -> Any:
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


def _bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "success", "recognizable"}:
            return True
        if lowered in {"false", "no", "n", "0", "failure", "unrecognizable"}:
            return False
    return default


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _failure_modes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _prompt(*, instruction: str, requested_letter: str, confidence_threshold: float) -> str:
    return f"""
You are judging the final result of a simulated marker-writing task with a
lenient human-recognition standard.

Task:
{instruction}

Requested letter: {requested_letter}

You are given one image: the post-rollout whiteboard view with black ink rendered
only from measured marker-tip contact samples. Judge only the black ink trace on
the whiteboard. Do not reward or penalize hidden robot state, subgoal tracking,
VLM waypoint choices, or any imagined target template. The question is whether a
human viewer could reasonably recognize the produced ink as the requested
letter, allowing for noisy robot handwriting.

Return JSON only:
{{
  "requested_letter": "{requested_letter}",
  "recognizable": true/false,
  "confidence": <0-1>,
  "reason": "<brief visual reason>",
  "failure_modes": ["<optional short labels>"]
}}

Use confidence >= {confidence_threshold:.2f} when the mark is reasonably
recognizable as the requested letter to a human viewer. Be lenient about
robot-writing artifacts: wavy strokes, uneven pressure, short or fragmented
secondary strokes, small gaps, and slight distortions should still be marked
recognizable when the intended letter is apparent. Set recognizable to false
only when the trace is mostly missing, has no plausible structure of the
requested letter, or is more clearly another character.
""".strip()


def _parse_response(
    data: Mapping[str, Any],
    *,
    requested_letter: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    recognized_letter = str(data.get("requested_letter") or requested_letter).strip().upper()
    confidence = _confidence(data.get("confidence"))
    recognizable = _bool(data.get("recognizable"), default=False)
    letter_matches = recognized_letter == requested_letter.upper()
    success = bool(recognizable and letter_matches and confidence >= float(confidence_threshold))
    return {
        "schema_version": "toolframe_marker_trace_vlm_judge_v1",
        "success": success,
        "recognizable": recognizable,
        "requested_letter": requested_letter.upper(),
        "vlm_requested_letter": recognized_letter,
        "letter_matches_request": letter_matches,
        "confidence": confidence,
        "confidence_threshold": float(confidence_threshold),
        "reason": str(data.get("reason") or ""),
        "failure_modes": _failure_modes(data.get("failure_modes")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--ink-overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--response-mode", default="openai-compatible")
    parser.add_argument("--model", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--requested-letter")
    args = parser.parse_args()

    task = _load_json(args.task)
    instruction = str(task.get("instruction") or "")
    requested_letter = str(args.requested_letter or "").strip().upper()
    if not requested_letter:
        match = re.search(r"\bletter\s+([A-Za-z])\b", instruction)
        if match:
            requested_letter = match.group(1).upper()
    if not re.fullmatch(r"[A-Z]", requested_letter):
        raise ValueError(
            "marker trace VLM judge requires one requested English letter; "
            f"got {requested_letter!r}"
        )
    if not args.ink_overlay.exists():
        raise FileNotFoundError(f"ink overlay does not exist: {args.ink_overlay}")

    prompt = _prompt(
        instruction=instruction,
        requested_letter=requested_letter,
        confidence_threshold=args.confidence_threshold,
    )
    client = build_vlm_client(
        response_mode=args.response_mode,
        model=args.model,
        image_paths=[args.ink_overlay],
        max_new_tokens=512,
    )
    response = client.generate(
        prompt=prompt,
        task_instruction=instruction,
        context_id=f"marker_trace_vlm_judge_{requested_letter.lower()}",
    )
    parsed_raw = json.loads(_strip_json_response(response.raw_text))
    result = _parse_response(
        parsed_raw,
        requested_letter=requested_letter,
        confidence_threshold=args.confidence_threshold,
    )
    result.update(
        {
            "task": str(args.task),
            "instruction": instruction,
            "ink_overlay": str(args.ink_overlay),
            "raw_response": response.raw_text,
            "vlm_metadata": response.metadata,
        }
    )
    _write_json(args.output, result)
    print(json.dumps({"vlm_judge_result": str(args.output), "success": result["success"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
