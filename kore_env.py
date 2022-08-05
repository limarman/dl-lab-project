import copy
from typing import Tuple, Union, Callable
import gym
import numpy as np
from gym.core import ActType, ObsType
from gym.vector.utils import spaces
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction

from src.Actions.action_adapter import ActionAdapter
from src.Rewards.kore_reward import KoreReward
from src.Environment.helpers import get_boards_from_kore_env_state, get_info_logs
from src.States.board_wrapper import BoardWrapper
import src.core.global_vars


class KoreEnv(gym.Env):
    ENV_NAME: str = "kore_fleets"

    def __init__(
            self,
            state_constr,  # eg hybrid state
            action_adapter: ActionAdapter,
            kore_reward: KoreReward,
            enemy_agent: Union[str, Callable] = 'balanced',
            opponent_agent_name=0,  # for logging we get it for selfplay_env
    ):

        self.env = None
        self.opponent_agent = None
        self.opponent_agent_name = opponent_agent_name
        self.action_adapter = action_adapter
        self.reward_calculator = kore_reward
        self.state_constr = state_constr
        self.enemy_agent = enemy_agent
        self.player_id = 0  # my player
        self.enemy_id = 1  # enemy player
        self.boards = {}
        self.current_state = None
        self.results = []
        self.replay = None
        self.reset()
        self.step_counter = 0
        self.shipyards = []
        self.shipyards_enemy = []
        self.shipyard_actions = {}
        self.shipyard_action_names = {}
        self.my_action_names = []  # for w&b histogram
        self.opponent_action_names = []  # for w&b histogram
        self.num_shipyards = 1
        self.reward = 0
        self.reward_info = {}
        self.current_shipyard = None,
        self.current_enemy_shipyard = None,
        self.current_shipyard_enemy = None
        self.current_action = None,
        self.current_action_name = None,
        self.last_game_step_board = None,
        self.game_step_flag = None
        self.anneal_time = 0
        self.non_zero_reward_count = 0

        # initialize the step counter
        src.core.global_vars.init_global_step()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # action is a list (len = N_ACTIONS) of ints (this comes from the NN)
        """
        The game step is divided into substeps for each Shipyard 1<=i<=number of shipyards.
        In each substep i, we determine the action for shipyard i and update the state by
        applying the action for shipyard 1 to i to the board.

        In step i = number of shipyards, we perform the actual game step
        """

        board_wrapper = BoardWrapper(self.boards[self.player_id], self.player_id)

        # the action prediction is for current shipyard
        if self.current_shipyard:  # for the first run this comes from the reset() function and later from the _game_step
            # current_action is of the type ShipyardAction
            # this is where we get a my_action in the form the kore expects (action from NN is fed to the step function)
            self.current_action, self.current_action_name = self.action_adapter.agent_to_kore_action(action,
                                                                                                     board_wrapper,
                                                                                                     self.current_shipyard)
        else:  # we have exhausted all the shipyards
            self.current_action, self.current_action_name = {}, {}

        self.my_action_names.append(str(self.current_action_name.values()))

        self.shipyard_actions.update(self.current_action)  # eg {'0-1': 'SPAWN_1'}
        self.shipyard_action_names.update(self.current_action_name)

        # self.shipyards is the list of shipyards that still need to be processed until the next game step
        if self.shipyards:  # checking if there are still some shipyards left
            self._shipyard_sub_step()  # simulate actions for all shipyards and update the board
            self.game_step_flag = False
        else:
            self._game_step()
            self.game_step_flag = True

        if self.shipyards:
            self.current_shipyard = self.shipyards.pop(0)
        else:
            # we have lost since we have no shipyards
            self.current_shipyard = None

        next_state = self.state_constr(self.boards[self.player_id], self.current_shipyard, recenter=False)

        if self.game_step_flag:
            self.reward, self.reward_info = self.reward_calculator.get_reward(self.current_state, next_state, None)
            self.anneal_time += 1
        else:
            self.reward = 0

        info = get_info_logs(next_state, self.current_action, self.current_action_name)
        # adding more information (E in advantage reward) to the info dict for logging purpose

        if 'E' in self.reward_info and isinstance(self.reward_info, dict):
            info = info | self.reward_info
        else:
            info = info | {'E': 0}

        info = info | {'enemy agent': self.opponent_agent_name} | {'my action name list': self.my_action_names} | {
            'enemy action name list': self.opponent_action_names}

        # print(f"Opponent_agent_name: {self.opponent_agent_name}")

        self.current_state = next_state

        if self.env.done:
            self.replay = self.render()

        # print("Opponent model:", self.opponent_agent_name)

        next_state_tensor = next_state.tensor

        # increment the global step count
        src.core.global_vars.increment_step_count()

        return next_state_tensor, self.reward, self.env.done, info

    def _shipyard_sub_step(self):
        """
        - Simulates all accumulated actions on the board and updates the board in-place
        """
        # probably too much copying since next() also makes a shallow copy
        # but better safe than sorry :)
        current_copy = copy.deepcopy(self.last_game_step_board)
        for shipyard in current_copy.current_player.shipyards:
            if shipyard.id in self.shipyard_actions:  # if there is an action for a shipyard
                action_object = ShipyardAction.from_str(self.shipyard_actions[shipyard.id])
                shipyard.next_action = action_object

        self.boards[self.player_id] = current_copy.next()

    def _game_step(self):
        """
        - Performs an actual game step, i.e. calls the kaggle actions with all accumulated
        shipyard actions and simulates the opponent
        - Updates our and the opponents board in-place
        - Collects new shipyards into our shipyard arrays
        """
        next_kore_action = [self.shipyard_actions]
        # this is the place where we get the opponent action
        next_opponent_action = self.opponent_agent(self.boards[1].observation, self.env.configuration)
        next_kore_action.append(next_opponent_action)

        step_result = self.env.step(next_kore_action)  # Env step NOT our step function defined above
        self.boards = get_boards_from_kore_env_state(step_result, self.env.configuration)

        self.shipyard_actions = {}
        self.shipyard_action_names = {}

        self.shipyards = self.boards[0].current_player.shipyards
        self.num_shipyards = len(self.shipyards)

        self.last_game_step_board = copy.deepcopy(self.boards[0])

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.env = make(self.ENV_NAME, debug=True)
        # setting the enemy agent (balanced bot by default)
        if callable(self.enemy_agent):
            self.opponent_agent = self.enemy_agent
        else:
            self.opponent_agent = self.env.agents[self.enemy_agent]
        init_step_result = self.env.reset(num_agents=2)
        self.boards = get_boards_from_kore_env_state(init_step_result, self.env.configuration)
        self.shipyards = self.boards[0].current_player.shipyards
        self.current_shipyard = self.shipyards.pop()

        self.shipyards_enemy = self.boards[1].current_player.shipyards
        self.current_shipyard_enemy = self.shipyards_enemy.pop()

        self.current_state = self.state_constr(self.boards[self.player_id], self.current_shipyard, recenter=True)

        self.reward_calculator.reset()

        return self.current_state.tensor

    def render(self, mode="html"):
        return self.env.render(mode="html")

    @property
    def observation_space(self):
        spaces = {
            'maps': gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=self.state_constr.get_input_shape()['maps']),
            'scalars': gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=self.state_constr.get_input_shape()['scalars'])
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return spaces.Box(low=0, high=1, shape=[self.action_adapter.N_ACTIONS])

    @property
    def num_envs(self):
        return 2
