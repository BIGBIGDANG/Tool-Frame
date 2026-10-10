"""Create the fixed-goal simulator from a policy config and shared overrides."""
# IsaacGym must be imported before torch (via SimToolReal).
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from deployment.rl_player_utils import read_cfg_omegaconf
from isaacgymenvs.utils.reformat import omegaconf_to_dict


def create_env(config_path, device, headless=False, enable_viewer_sync_at_start=True,
               episode_length=None, overrides=None):
    policy_cfg = read_cfg_omegaconf(config_path, device)
    config_dir = Path(__file__).resolve().parents[2] / "isaacgymenvs/cfg"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        defaults = compose(config_name="config")
    OmegaConf.set_struct(defaults, False)
    OmegaConf.set_struct(policy_cfg, False)
    cfg = OmegaConf.merge(defaults, policy_cfg)
    cfg.headless = headless
    cfg.task.sim.enable_viewer_sync_at_start = enable_viewer_sync_at_start
    cfg.task.env.numEnvs = 1
    if episode_length is not None:
        cfg.task.env.episodeLength = episode_length
    cfg.graphics_device_id = int(device.split(":")[-1]) if device.startswith("cuda:") else 0
    for key, value in (overrides or {}).items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    return SimToolReal(cfg=omegaconf_to_dict(cfg.task), sim_device=device, rl_device=device,
                      graphics_device_id=cfg.graphics_device_id, headless=headless,
                      virtual_screen_capture=False, force_render=True)
