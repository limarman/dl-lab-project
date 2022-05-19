from typing import Tuple, Union

import gym
from gym.core import ActType, ObsType
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import Board
from kaggle_environments.envs.kore_fleets.kore_fleets import balanced_agent

from src.Actions.action_adapter import ActionAdapter
from src.Rewards.kore_reward import KoreReward
from src.Environment.helpers import get_boards_from_kore_env_state


class KoreEnv(gym.Env):
    ENV_NAME: str = "kore_fleets"

    def __init__(self, state_constr, action_adapter: ActionAdapter, reward_calculator: KoreReward, agent: str ='balanced'):
        self.env = make(self.ENV_NAME, debug=True)
        self.action_adapter = ActionAdapter()
        self.reward_calculator = reward_calculator
        self.state_constr = state_constr
        self.two_player = bool(agent)
        self.opponent_agent = self.env.agents[agent]
        self.player_id = 0
        self.boards = {}
        self.current_state = None
        # set initial state and boards
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        next_kore_action = [self.action_adapter.agent_to_kore_action(action)]
        if self.two_player:
            next_opponent_action = self.opponent_agent(self.boards[1].observation, self.env.configuration)
            next_kore_action.append(next_opponent_action)

        step_result = self.env.step(next_kore_action)
        self.boards = get_boards_from_kore_env_state(step_result, self.env.configuration)
        next_state = self.state_constr(self.boards[self.player_id])
        next_reward = self.reward_calculator.get_reward_from_states(self.current_state, next_state)

        info = {}
        self.current_state = next_state

        return next_state.tensor, next_reward, self.env.done, info

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        init_step_result = self.env.reset(num_agents=2)
        self.boards = get_boards_from_kore_env_state(init_step_result, self.env.configuration)
        self.current_state = self.state_constr(self.boards[self.player_id])

        return self.current_state.tensor

    def render(self, mode="html", close=False):
        return self.env.render(mode=mode, width=1000, height=800)
