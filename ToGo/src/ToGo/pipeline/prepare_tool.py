"""P3SAM segmentation and conversion into the prepared tool-prior contract."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ToGo.pipeline.handoff import ROOT, write_json
from ToGo.providers.tool_prior.render_cache import ensure_render_cache
from ToGo.providers.tool_prior.sampart3d_mesh_candidates import build_sampart3d_mesh_tool_prior


def prepare_tool(*, tool_id: str, mesh: Path, output: Path, face_labels: Path | None,
                 p3sam_root: Path | None, p3sam_python: str) -> None:
    mesh = mesh.resolve()
    tool_root = output.resolve() / tool_id
    copied_mesh = tool_root / "mesh_root" / mesh.name
    copied_mesh.parent.mkdir(parents=True, exist_ok=True)
    if mesh != copied_mesh:
        shutil.copy2(mesh, copied_mesh)
        for suffix in (".mtl", "_visual.glb"):
            source = mesh.with_suffix(suffix) if suffix == ".mtl" else mesh.with_name(mesh.stem + suffix)
            if source.exists():
                shutil.copy2(source, copied_mesh.parent / source.name)
        if (mesh.parent / "material.mtl").exists():
            shutil.copy2(mesh.parent / "material.mtl", copied_mesh.parent / "material.mtl")
        if (mesh.parent / "textures").is_dir():
            shutil.copytree(mesh.parent / "textures", copied_mesh.parent / "textures", dirs_exist_ok=True)
    if face_labels is None:
        if p3sam_root is None:
            raise ValueError("provide --p3sam-root or existing --face-labels")
        p3sam_root = p3sam_root.resolve()
        raw = tool_root / "segmentation"
        subprocess.run([p3sam_python, str(p3sam_root / "demo/auto_mask.py"),
            "--ckpt_path", str(p3sam_root / "weights/p3sam/p3sam.safetensors"),
            "--mesh_path", str(copied_mesh), "--output_path", str(raw),
            "--point_num", "100000", "--prompt_num", "400", "--prompt_bs", "8",
            "--threshold", "0.95", "--seed", "42", "--parallel", "0", "--clean_mesh", "0"],
            cwd=p3sam_root / "demo", check=True,
            env={**os.environ, "PYTHONPATH": str(p3sam_root) + os.pathsep + os.environ.get("PYTHONPATH", "")})
        choices = [raw / name for name in ("auto_mask_mesh_final_face_ids.npy",
                   "auto_mask_mesh_filtered_2_face_ids.npy", "auto_mask_mesh_face_ids.npy", "auto_mask_mesh_org_face_ids.npy")]
        face_labels = next((path for path in choices if path.is_file()), None)
        if face_labels is None:
            raise FileNotFoundError("P3SAM did not produce mesh face labels")
    labels = np.load(face_labels, allow_pickle=False)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("face labels must be a 1D integer array matching the original mesh faces")
    label_path = tool_root / "face_labels/p3sam_face_labels.npy"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(label_path, labels.astype(np.int64))
    artifacts = build_sampart3d_mesh_tool_prior(tool_id=tool_id, source_mesh=copied_mesh,
        face_labels_path=label_path, observation_id=tool_id, min_face_count=1,
        segmentation_mode="p3sam", provider_id="p3sam_tool_prior_provider")

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and value.startswith(str(ROOT) + os.sep):
            return str(Path(value).relative_to(ROOT))
        return value

    for name in ("tool_part_graph", "grounding_candidates"):
        data = artifacts[name]
        write_json(tool_root / "tool_prior" / (name + ".json"), portable(data))
    ensure_render_cache(tool_root, "tool_prior")
