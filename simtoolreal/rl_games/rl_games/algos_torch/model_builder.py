"""Model construction for the continuous SimToolReal actor-critic checkpoint."""
from rl_games.algos_torch import network_builder, models

class ModelBuilder:
    def load(self, params):
        if params["model"]["name"] != "continuous_a2c_logstd" or params["network"]["name"] != "actor_critic":
            raise ValueError("unsupported SimToolReal policy architecture")
        network = network_builder.A2CBuilder()
        network.load(params["network"])
        return models.ModelA2CContinuousLogStd(network)
