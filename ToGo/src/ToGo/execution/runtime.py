"""IsaacGym compatibility and scene rendering helpers; no task scoring."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Sequence
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = Path(os.environ.get("TOGO_PRETRAINED_POLICY_ROOT", str(ROOT / "pretrained_policy"))) / "config.yaml"
DEFAULT_RENDER_LIGHT_PRESET = "soft"
DEFAULT_SOFT_LIGHT_INDICES = (0, 1)
DEFAULT_SOFT_LIGHT_INTENSITY = (0.45, 0.45, 0.45)
DEFAULT_SOFT_LIGHT_AMBIENT = (0.78, 0.78, 0.78)
DEFAULT_SOFT_LIGHT_DIRECTION = (0.5, -0.8, 1.0)

def scene_actor_specs(state_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    actors = state_bundle.get("scene_actors", [])
    if not isinstance(actors, list) or any(not isinstance(actor, dict) for actor in actors):
        raise ValueError("state_bundle.scene_actors must be a list of actor objects")
    return actors

def apply_simtoolreal_render_lighting(
    env: Any,
    preset: str = DEFAULT_RENDER_LIGHT_PRESET,
    *,
    soft_light_indices: Sequence[int] = DEFAULT_SOFT_LIGHT_INDICES,
) -> dict[str, Any]:
    """Apply the shared render lighting preset to a SimToolReal env."""
    if preset == "native":
        return {"preset": "native", "uses_simtoolreal_default": True}
    if preset != "soft":
        raise ValueError(f"unsupported light preset {preset!r}")
    from isaacgym import gymapi  # noqa: WPS433

    light_indices = tuple(int(item) for item in soft_light_indices)
    for light_index in light_indices:
        env.gym.set_light_parameters(
            env.sim,
            light_index,
            gymapi.Vec3(*DEFAULT_SOFT_LIGHT_INTENSITY),
            gymapi.Vec3(*DEFAULT_SOFT_LIGHT_AMBIENT),
            gymapi.Vec3(*DEFAULT_SOFT_LIGHT_DIRECTION),
        )
    return {
        "preset": "soft",
        "name": (
            "soft01"
            if light_indices == DEFAULT_SOFT_LIGHT_INDICES
            else ("soft02" if light_indices == (0, 2) else "soft_custom")
        ),
        "light_indices": list(light_indices),
        "intensity": list(DEFAULT_SOFT_LIGHT_INTENSITY),
        "ambient": list(DEFAULT_SOFT_LIGHT_AMBIENT),
        "direction": list(DEFAULT_SOFT_LIGHT_DIRECTION),
        "applied_to": "simtoolreal_env",
        "note": "local IsaacGym binding exposes intensity, ambient, and direction",
    }


def _install_numpy_legacy_aliases() -> None:
    """Restore NumPy aliases expected by older robotics dependencies."""

    import numpy as np  # noqa: WPS433
    import warnings  # noqa: WPS433

    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
    }.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            has_alias = hasattr(np, name)
        if not has_alias:
            setattr(np, name, value)
