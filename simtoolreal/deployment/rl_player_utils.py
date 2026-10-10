from typing import Optional
import yaml
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict

def read_cfg(config_path: str, device: Optional[str] = None) -> dict:
    # Read in config
    # Change device if needed

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Convert to OmegaConf to process interpolations
    omegaconf_cfg = OmegaConf.create(raw_cfg)

    # Need to convert back to dict
    cfg = omegaconf_to_dict(omegaconf_cfg)

    # Modify device manually
    if device is not None:
        if (
            "train" in cfg
            and "params" in cfg["train"]
            and "config" in cfg["train"]["params"]
        ):
            # OLD
            cfg["train"]["params"]["config"]["device"] = device
            cfg["train"]["params"]["config"]["device_name"] = device
        else:
            # NEW
            cfg["rl_device"] = device
            cfg["sim_device"] = device


    return cfg


def read_cfg_omegaconf(config_path: str, device: Optional[str] = None) -> DictConfig:
    # Read in config
    # Change device if needed

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Convert to OmegaConf to process interpolations
    omegaconf_cfg = OmegaConf.create(raw_cfg)

    # Modify device manually
    if device is not None:
        if (
            "train" in omegaconf_cfg
            and "params" in omegaconf_cfg.train
            and "config" in omegaconf_cfg.train.params
        ):
            # OLD
            omegaconf_cfg.train.params.config.device = device
            omegaconf_cfg.train.params.config.device_name = device
            omegaconf_cfg.rl_device = device
            omegaconf_cfg.sim_device = device
        else:
            # NEW
            omegaconf_cfg.rl_device = device
            omegaconf_cfg.sim_device = device


    return omegaconf_cfg
