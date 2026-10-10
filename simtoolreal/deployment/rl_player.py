"""Inference for the SimToolReal continuous LSTM policy used by ToGo 6.13."""
from copy import deepcopy
import torch
from deployment.rl_player_utils import read_cfg
from rl_games.algos_torch.model_builder import ModelBuilder


class RlPlayer:
    def __init__(self, num_observations, num_actions, config_path, checkpoint_path,
                 device, num_envs=1):
        self.num_observations = int(num_observations)
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        params = deepcopy(read_cfg(config_path, device)["train"]["params"])
        config = params["config"]
        if params["model"]["name"] != "continuous_a2c_logstd" or params["network"]["name"] != "actor_critic":
            raise ValueError("expected the ToGo 6.13 continuous actor-critic policy")
        if config.get("expl_type") != "mixed_expl_learn_param":
            raise ValueError("expected the SimToolReal policy's learned coefficient input")
        network = ModelBuilder().load(params)
        # Preserve the pretrained policy's extra scalar observation and coefficient table.
        model_config = {"actions_num": int(num_actions), "input_shape": (self.num_observations + 1,),
                        "num_seqs": self.num_envs, "value_size": 1,
                        "normalize_value": config.get("normalize_value", False),
                        "normalize_input": config["normalize_input"], "type": "extra_param",
                        "coef_ids": torch.linspace(50.0, 0.0, 6, device=self.device),
                        "coef_id_idx": self.num_observations}
        self.model = network.build(model_config).to(self.device)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(payload, list) or (isinstance(payload, dict) and 0 in payload):
            payload = payload[0]
        self.model.load_state_dict(payload["model"])
        if config["normalize_input"] and "running_mean_std" in payload:
            self.model.running_mean_std.load_state_dict(payload["running_mean_std"])
        self.environment_state = payload.get("env_state")
        self.model.eval()
        self.states = [state.to(self.device) for state in self.model.get_default_rnn_state()] if self.model.is_rnn() else None
        self.clip_actions = config.get("clip_actions", True)

    def get_normalized_action(self, obs, deterministic_actions=True):
        if tuple(obs.shape) != (self.num_envs, self.num_observations):
            raise ValueError("policy observation dimensions differ from the pretrained protocol")
        obs = torch.cat([obs, torch.full((self.num_envs, 1), 50.0, device=self.device)], dim=1)
        with torch.no_grad():
            result = self.model({"is_train": False, "prev_actions": None,
                                 "obs": obs, "rnn_states": self.states})
        self.states = result["rnn_states"]
        action = result["mus"] if deterministic_actions else result["actions"]
        return action.clamp(-1.0, 1.0) if self.clip_actions else action
