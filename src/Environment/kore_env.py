from typing import Tuple, Union, Callable
import gym
import numpy as np
from gym.core import ActType, ObsType
from gym.vector.utils import spaces
from kaggle_environments import make

from src.Actions.action_adapter import ActionAdapter
from src.Actions.action_adapter_rule_based import RuleBasedActionAdapter
from src.Rewards.kore_reward import KoreReward
from src.Environment.helpers import get_boards_from_kore_env_state, get_info_logs
from src.States.board_wrapper import BoardWrapper


class KoreEnv(gym.Env):
    ENV_NAME: str = "kore_fleets"

    def __init__(
            self,
            state_constr,
            action_adapter: ActionAdapter,
            kore_reward: KoreReward,
            enemy_agent: Union[str, Callable] = 'balanced'
    ):
        self.env = None
        self.opponent_agent = None
        self.action_adapter = action_adapter
        self.reward_calculator = kore_reward
        self.state_constr = state_constr
        self.enemy_agent = enemy_agent
        self.player_id = 0
        self.boards = {}
        self.current_state = None
        self.results = []
        self.replay = None
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        board_wrapper = BoardWrapper(self.boards[self.player_id], self.player_id)
        actions_me, action_names_me = self.action_adapter.agent_to_kore_action(action, board_wrapper)

        next_kore_action = [actions_me]
        next_opponent_action = self.opponent_agent(self.boards[1].observation, self.env.configuration)
        next_kore_action.append(next_opponent_action)

        step_result = self.env.step(next_kore_action)
        self.boards = get_boards_from_kore_env_state(step_result, self.env.configuration)
        next_state = self.state_constr(self.boards[self.player_id])
        next_reward = self.reward_calculator.get_reward(self.current_state, next_state, next_kore_action[0])

        info = get_info_logs(next_state, actions_me, action_names_me)

        self.current_state = next_state
        if self.env.done:
            self.replay = self.render()

        return next_state.tensor, next_reward, self.env.done, info

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.env = make(self.ENV_NAME, debug=True)
        if callable(self.enemy_agent):
            self.opponent_agent = self.enemy_agent
        else:
            self.opponent_agent = self.env.agents[self.enemy_agent]
        init_step_result = self.env.reset(num_agents=2)
        self.boards = get_boards_from_kore_env_state(init_step_result, self.env.configuration)
        self.current_state = self.state_constr(self.boards[self.player_id])

        return self.current_state.tensor

    def render(self, mode="html"):
        return self.env.render(mode="html")#, width=500, height=6600)

    @property
    def observation_space(self):
        spaces = {
            'maps': gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=self.state_constr.get_input_shape()['maps']),
            'scalars': gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=self.state_constr.get_input_shape()['scalars'])
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return spaces.Box(low=0, high=1, shape=[RuleBasedActionAdapter.N_ACTIONS])

    @property
    def num_envs(self):
        return 2
