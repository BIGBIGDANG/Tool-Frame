#!/usr/bin/env python3
"""Build clean ToolFrame tool refs from a P3SAM mesh segmentation.

This is the P3SAM-first tool-side entry point for the image-grounded mainline:

1. optionally run Hunyuan3D-Part/P3-SAM auto_mask.py on a tool mesh;
2. convert the raw P3SAM face-id output into a 1-D face-label array;
3. reuse the generic mesh-label tool-prior builder to emit ToolPartGraph,
   FunctionalToolParts, GroundingCandidates, ProviderContract, and Manifest.

No task primitive, tool family, or repair-loop state is introduced here. The
output is only geometry refs in the stable SimToolReal tool frame.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parttoolgen.providers.tool_prior.sampart3d_tool_prior_provider import (  # noqa: E402
    convert_sampart3d_label_file,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_P3SAM_ROOT = Path(
    os.environ.get(
        "PARTTOOLGEN_P3SAM_ROOT",
        str(REPO_ROOT / "artifacts" / "models" / "Hunyuan3D-Part" / "P3-SAM"),
    )
)
DEFAULT_CKPT = DEFAULT_P3SAM_ROOT / "weights/p3sam/p3sam.safetensors"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _candidate_face_id_files(raw_dir: Path) -> list[Path]:
    preferred = [
        raw_dir / "auto_mask_mesh_final_face_ids.npy",
        raw_dir / "auto_mask_mesh_filtered_2_face_ids.npy",
        raw_dir / "auto_mask_mesh_face_ids.npy",
        raw_dir / "auto_mask_mesh_org_face_ids.npy",
    ]
    seen: set[Path] = set()
    files: list[Path] = []
    for path in preferred + sorted(raw_dir.glob("*face_ids.npy")):
        if path.exists() and path not in seen:
            files.append(path)
            seen.add(path)
    return files


def _convert_p3sam_face_ids(
    *,
    raw_dir: Path,
    output_path: Path,
    unlabeled_value: int = -1,
) -> dict[str, Any]:
    candidates = _candidate_face_id_files(raw_dir)
    if not candidates:
        raise FileNotFoundError(f"no P3SAM *face_ids.npy found under {raw_dir}")
    source = candidates[0]
    labels = np.load(source, allow_pickle=False)
    if labels.ndim != 1:
        raise ValueError(f"expected 1-D P3SAM face ids in {source}, got shape {labels.shape}")
    labels = labels.astype(np.int64, copy=True)
    labels[labels < 0] = int(unlabeled_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "source_face_ids": str(source),
        "output_face_labels": str(output_path),
        "face_count": int(labels.shape[0]),
        "unlabeled_value": int(unlabeled_value),
        "unique_labels": [int(item) for item in unique.tolist()],
        "label_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
    }


def _run_p3sam(
    *,
    python: Path,
    p3sam_root: Path,
    ckpt_path: Path,
    mesh_path: Path,
    raw_dir: Path,
    point_num: int,
    prompt_num: int,
    prompt_bs: int,
    threshold: float,
    seed: int,
    parallel: int,
    clean_mesh: int,
) -> dict[str, Any]:
    auto_mask = p3sam_root / "demo/auto_mask.py"
    if not auto_mask.exists():
        raise FileNotFoundError(f"P3SAM auto_mask.py not found: {auto_mask}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"P3SAM checkpoint not found: {ckpt_path}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(auto_mask),
        "--ckpt_path",
        str(ckpt_path),
        "--mesh_path",
        str(mesh_path),
        "--output_path",
        str(raw_dir),
        "--point_num",
        str(int(point_num)),
        "--prompt_num",
        str(int(prompt_num)),
        "--prompt_bs",
        str(int(prompt_bs)),
        "--threshold",
        str(float(threshold)),
        "--seed",
        str(int(seed)),
        "--parallel",
        str(int(parallel)),
        "--clean_mesh",
        str(int(clean_mesh)),
    ]
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            str(p3sam_root)
            if not env.get("PYTHONPATH")
            else str(p3sam_root) + os.pathsep + env["PYTHONPATH"]
        )
        result = subprocess.run(
            command,
            cwd=str(p3sam_root / "demo"),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stdout or "")[-8000:]
        raise RuntimeError(f"P3SAM auto_mask failed with exit code {exc.returncode}:\n{tail}") from exc
    return {
        "command": command,
        "cwd": str(p3sam_root / "demo"),
        "stdout_tail": result.stdout[-8000:],
    }


def build_p3sam_tool_prior(
    *,
    tool_id: str,
    source_mesh: Path,
    output_root: Path,
    p3sam_root: Path,
    ckpt_path: Path,
    python: Path,
    skip_p3sam_run: bool,
    p3sam_raw_dir: Path | None,
    point_num: int,
    prompt_num: int,
    prompt_bs: int,
    threshold: float,
    seed: int,
    parallel: int,
    clean_mesh: int,
    min_face_count: int,
    observation_id: str | None,
) -> dict[str, Any]:
    source_mesh = source_mesh.resolve()
    output_root = output_root.resolve()
    p3sam_root = p3sam_root.resolve()
    ckpt_path = ckpt_path.resolve()
    python = Path(python)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir = (p3sam_raw_dir.resolve() if p3sam_raw_dir is not None else output_root / "p3sam_raw")
    run_report: dict[str, Any] | None = None
    if not skip_p3sam_run:
        run_report = _run_p3sam(
            python=python,
            p3sam_root=p3sam_root,
            ckpt_path=ckpt_path,
            mesh_path=source_mesh,
            raw_dir=raw_dir,
            point_num=point_num,
            prompt_num=prompt_num,
            prompt_bs=prompt_bs,
            threshold=threshold,
            seed=seed,
            parallel=parallel,
            clean_mesh=clean_mesh,
        )
    elif not raw_dir.exists():
        raise FileNotFoundError(f"--skip-p3sam-run was set but raw dir does not exist: {raw_dir}")

    labels_path = output_root / "face_labels/p3sam_face_labels.npy"
    conversion_report = _convert_p3sam_face_ids(
        raw_dir=raw_dir,
        output_path=labels_path,
    )
    tool_prior_dir = output_root / "tool_prior"
    convert_sampart3d_label_file(
        tool_id=tool_id,
        source_mesh=source_mesh,
        face_labels_path=labels_path,
        output_dir=tool_prior_dir,
        observation_id=observation_id or f"{tool_id}_p3sam_auto_mask",
        min_face_count=min_face_count,
        segmentation_mode="p3sam",
        provider_id="p3sam_tool_prior_provider",
    )
    mesh_copy = output_root / "mesh_root" / source_mesh.name
    mesh_copy.parent.mkdir(parents=True, exist_ok=True)
    if source_mesh.resolve() != mesh_copy.resolve():
        shutil.copy2(source_mesh, mesh_copy)

    summary = {
        "tool_id": tool_id,
        "source_mesh": str(source_mesh),
        "output_root": str(output_root),
        "p3sam_raw_dir": str(raw_dir),
        "face_labels": str(labels_path),
        "tool_prior_dir": str(tool_prior_dir),
        "p3sam_run": run_report,
        "face_label_conversion": conversion_report,
        "parameters": {
            "skip_p3sam_run": bool(skip_p3sam_run),
            "point_num": int(point_num),
            "prompt_num": int(prompt_num),
            "prompt_bs": int(prompt_bs),
            "threshold": float(threshold),
            "seed": int(seed),
            "parallel": int(parallel),
            "clean_mesh": int(clean_mesh),
            "min_face_count": int(min_face_count),
            "observation_id": observation_id or f"{tool_id}_p3sam_auto_mask",
        },
    }
    _write_json(output_root / "p3sam_tool_prior_build_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-id", required=True)
    parser.add_argument("--source-mesh", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--p3sam-root", default=DEFAULT_P3SAM_ROOT, type=Path)
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT, type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument("--skip-p3sam-run", action="store_true")
    parser.add_argument("--p3sam-raw-dir", type=Path)
    parser.add_argument("--point-num", type=int, default=100000)
    parser.add_argument("--prompt-num", type=int, default=400)
    parser.add_argument("--prompt-bs", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--clean-mesh", type=int, default=1)
    parser.add_argument("--min-face-count", type=int, default=1)
    parser.add_argument("--observation-id")
    args = parser.parse_args()
    summary = build_p3sam_tool_prior(
        tool_id=args.tool_id,
        source_mesh=args.source_mesh,
        output_root=args.output_root,
        p3sam_root=args.p3sam_root,
        ckpt_path=args.ckpt_path,
        python=args.python,
        skip_p3sam_run=args.skip_p3sam_run,
        p3sam_raw_dir=args.p3sam_raw_dir,
        point_num=args.point_num,
        prompt_num=args.prompt_num,
        prompt_bs=args.prompt_bs,
        threshold=args.threshold,
        seed=args.seed,
        parallel=args.parallel,
        clean_mesh=args.clean_mesh,
        min_face_count=args.min_face_count,
        observation_id=args.observation_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
