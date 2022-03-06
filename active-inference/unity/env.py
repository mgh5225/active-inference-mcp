import enum
import os
import numpy as np

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class EngineType(enum.Enum):
    Front = 0.0
    Rear = 1.0
    Both = 2.0


class Environment:

    def __init__(self, use_editor: bool = False, time_scale: float = 1) -> None:
        if use_editor:
            self.env_path = None
        else:
            self.env_path = os.path.join(
                os.path.dirname(__file__), "../../unity/Build/environment.x86_64")

        self.time_scale = time_scale
        self.env = None
        self.environment_parameters_channel = None
        self.behavior_name = None
        self.spec = None

    def init_env(self):
        engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()
        self.env = UnityEnvironment(file_name=self.env_path, base_port=5004,
                                    side_channels=[engine_configuration_channel, self.environment_parameters_channel])

        self.env.reset()

        self.behavior_name = list(self.env.behavior_specs)[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        engine_configuration_channel.set_configuration_parameters(
            time_scale=self.time_scale)

    def set_engine_type(self, type: EngineType):
        self.environment_parameters_channel.set_float_parameter(
            "engineType", type.value)

    def get_position(self):
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return decision_steps.obs[0][:, :2]

    def get_action(self):
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return decision_steps.obs[0][:, 2]

    def get_distance(self):
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return decision_steps.obs[0][:, 3:5]

    def set_action(self, action: float):
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.array([[action]]))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()
