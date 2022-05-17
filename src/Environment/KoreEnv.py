from typing import Tuple, Union

import gym
from gym.core import ActType, ObsType
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import Board

from src.Actions.ActionAdapter import ActionAdapter
from src.Monitoring.KoreMonitor import KoreMonitor
from src.States.StateAdapter import StateAdapter


class KoreEnv(gym.Env):

    ENV_NAME: str = "kore_fleets"

    def __init__(self, state_adapter: StateAdapter, action_adapter: ActionAdapter):
        self.env = make(self.ENV_NAME, debug=True)
        self.state_adapter = state_adapter
        self.action_adapter = ActionAdapter()

        self.current_kore: float = 0.0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        next_kore_action = self.action_adapter.agent_to_kore_action(action)
        step_result = self.env.step([next_kore_action])
        observation = step_result[0]["observation"]
        board = Board(observation, self.env.configuration)
        next_state = self.state_adapter.board_to_state(board)

        previous_kore = self.current_kore
        self.current_kore = board.current_player.kore
        kore_difference = self.current_kore - previous_kore
        next_reward = max(kore_difference, 0)
        info = {}

        return next_state.values, next_reward, self.env.done, info

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        init_step_result = self.env.reset(num_agents=1)
        init_obs = init_step_result[0]["observation"]
        board = Board(init_obs, self.env.configuration)
        self.current_kore = board.current_player.kore
        init_state = self.state_adapter.board_to_state(board)

        return init_state.values

    def render(self, mode="html", close=False):
        return self.env.render(mode=mode, width=1000, height=800)
