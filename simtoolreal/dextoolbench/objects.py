"""Physical tool assets and the grasp dimensions expected by the supplied policy."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from isaacgymenvs.utils.utils import get_repo_root_dir

@dataclass
class Object:
    urdf_path: Path
    scale: Tuple[float, float, float]
    need_vhacd: bool

    def __post_init__(self):
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"missing tool asset: {self.urdf_path}")


def rescale_by_factor(scale, factor):
    # This encodes grasp dimensions for the policy; the mesh stays in meters.
    return tuple(value * factor for value in scale)

NAME_TO_OBJECT = {
    'mallet_hammer': Object(urdf_path=get_repo_root_dir() / 'assets/urdf/dextoolbench/hammer/mallet_hammer/mallet_hammer.urdf', scale=rescale_by_factor((0.24, 0.03, 0.02), factor=25), need_vhacd=True),
    'claw_hammer': Object(urdf_path=get_repo_root_dir() / 'assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf', scale=rescale_by_factor((0.1, 0.0225, 0.015), factor=25), need_vhacd=True),
    'long_screwdriver': Object(urdf_path=get_repo_root_dir() / 'assets/urdf/dextoolbench/screwdriver/long_screwdriver/long_screwdriver.urdf', scale=rescale_by_factor((0.1, 0.03, 0.03), factor=25), need_vhacd=True),
    'short_screwdriver': Object(urdf_path=get_repo_root_dir() / 'assets/urdf/dextoolbench/screwdriver/short_screwdriver/short_screwdriver.urdf', scale=rescale_by_factor((0.07, 0.035, 0.035), factor=25), need_vhacd=True),
    'handle_eraser': Object(urdf_path=get_repo_root_dir() / 'assets/urdf/dextoolbench/eraser/handle_eraser/handle_eraser.urdf', scale=rescale_by_factor((0.09, 0.032, 0.01), factor=25), need_vhacd=True),
    'staples_marker': Object(urdf_path=get_repo_root_dir() / 'assets/urdf/dextoolbench/marker/staples_marker/staples_marker.urdf', scale=rescale_by_factor((0.12, 0.018, 0.018), factor=25), need_vhacd=True),
    'red_brush': Object(urdf_path=get_repo_root_dir() / 'assets/urdf/dextoolbench/brush/red_brush/red_brush.urdf', scale=rescale_by_factor((0.1, 0.02, 0.015), factor=25), need_vhacd=True),
    'blue_brush': Object(urdf_path=get_repo_root_dir() / 'assets/urdf/dextoolbench/brush/blue_brush/blue_brush.urdf', scale=rescale_by_factor((0.12, 0.035, 0.02), factor=25), need_vhacd=True),
}
