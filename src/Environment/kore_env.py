from typing import Tuple, Union

import gym
from gym.core import ActType, ObsType
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import Board

from src.Actions.action_adapter import ActionAdapter
from src.Rewards.kore_reward import KoreReward


class KoreEnv(gym.Env):
    ENV_NAME: str = "kore_fleets"

    def __init__(self, state_constr, action_adapter: ActionAdapter, reward_calculator: KoreReward):
        self.env = make(self.ENV_NAME, debug=True)
        self.action_adapter = ActionAdapter()
        self.reward_calculator = reward_calculator
        self.state_constr = state_constr
        # TODO make num_agents a param
        # set initial state
        self.env.reset(num_agents=1)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        next_kore_action = self.action_adapter.agent_to_kore_action(action)
        step_result = self.env.step([next_kore_action])
        observation = step_result[0]["observation"]
        board = Board(observation, self.env.configuration)
        next_state = self.state_constr(board)
        next_reward = self.reward_calculator.get_reward_from_states(self.current_state, next_state)
        info = {}
        self.current_state = next_state

        return next_state.tensor, next_reward, self.env.done, info

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        init_step_result = self.env.reset(num_agents=1)
        init_obs = init_step_result[0]["observation"]
        board = Board(init_obs, self.env.configuration)
        self.current_state = self.state_constr(board)

        return self.current_state.tensor

    def render(self, mode="html", close=False):
        return self.env.render(mode=mode, width=1000, height=800)
