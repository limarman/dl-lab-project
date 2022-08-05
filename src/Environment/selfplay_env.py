import os
import random
import time
import copy
import gym
import numpy as np

from stable_baselines3 import PPO
from .kore_env import KoreEnv
from src.States.board_wrapper import BoardWrapper
from src.Environment.helpers import get_boards_from_kore_env_state
from src.Actions.action_adapter import ActionAdapter
from src.Rewards.kore_reward import KoreReward
from typing import Tuple, Union, Callable
from gym.core import ActType, ObsType
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from src.Environment.helpers import get_boards_from_kore_env_state, get_info_logs
import src.core.global_vars


class SelfPlayEnv(KoreEnv):
    """Wrapper over the KoreEnv with the additional ability to execute self play """

    def __init__(self,
                 state_constr,  # eg hybrid state
                 action_adapter: ActionAdapter,
                 kore_reward: KoreReward,
                 run_id,
                 enemy_agent: Union[str, Callable] = 'balanced',
                 ):

        self.self_play_window = 10  # size of replay model memory
        self.opponent_agent_name = None
        self.opponent_agent = enemy_agent
        self.enemy_models = None
        self.enemy_model_names = None
        self.enemy_models_dict = None
        self.info = None
        self.done = None
        self.render_ = None
        self.next_state = None
        self.run_id = run_id
        self.model_path = "selfplay_models"
        self.model_name = "PPO"

        # creating a directory with the name of run_id for self play (if it does not already exist)
        path = os.path.join(os.path.abspath("../selfplay_models/"), self.model_name, self.run_id)

        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except:
            pass

        # self.enemy_models = self._load_all_models()
        self.enemy_models_dict = self._load_all_models()

        super(SelfPlayEnv, self).__init__(state_constr, action_adapter, kore_reward, enemy_agent=self.opponent_agent,
                                          opponent_agent_name=self.opponent_agent_name)

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
        if self.shipyards or self.shipyards_enemy:  # checking if there are still some shipyards left
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

        if self.shipyards_enemy:
            self.current_shipyard_enemy = self.shipyards_enemy.pop(0)
        else:
            # we have lost since we have no shipyards
            self.current_shipyard_enemy = None

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

        # print(f"opponent_agent_name: {self.opponent_agent_name}")

        self.current_state = next_state

        if self.env.done:
            self.replay = self.render()

        # increment the global step count
        src.core.global_vars.increment_step_count()

        # print("Opponent model:", self.opponent_agent_name)

        return next_state.tensor, self.reward, self.env.done, info

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.current_state = super(SelfPlayEnv, self).reset()

        # print("RESET")

        self.setup_enemy()

        return self.current_state

    def render(self, mode="html"):
        self.render_ = super(SelfPlayEnv, self).render(mode)

        return self.render_

    def _game_step(self):
        """
                - Performs an actual game step, i.e. calls the kaggle actions with all accumulated
                shipyard actions and simulates the opponent
                - Updates our and the opponents board in-place
                - Collects new shipyards into our shipyard arrays
                """
        next_kore_action = [self.shipyard_actions]
        # this is the place where we get the opponent action
        # TODO: Add a way to pick opponent actions by the same NN pipeline as we pick our actions
        if hasattr(self.opponent_agent, 'predict'):  # output from NN
            # print("USING NN")
            # print("self.opponent_agent: ", self.opponent_agent)
            board_wrapper = BoardWrapper(self.boards[self.enemy_id], 0)
            # for current_shipyard_enemy in self.shipyards_enemy:

            state = self.state_constr(self.boards[self.enemy_id], self.current_shipyard_enemy, recenter=True)
            # print("enemy shipyard ship count: ", self.current_shipyard_enemy.ship_count)

            if self.current_shipyard_enemy:  # if there is any enemy shipyard, only then take the action...
                enemy_action, _ = self.opponent_agent.predict(state.tensor)
                next_opponent_action, next_opponent_action_name = self.action_adapter.agent_to_kore_action(enemy_action,
                                                                                                           board_wrapper,
                                                                                                           self.current_shipyard_enemy)
            else:
                next_opponent_action, next_opponent_action_name = {}, {}

            next_kore_action.append(next_opponent_action)
            # print("next_opponent_action_name: ", next_opponent_action_name)
            self.opponent_action_names.append(str(next_opponent_action_name.values()))  # w&b login purposes

        elif callable(self.opponent_agent):  # balanced agent
            next_opponent_action = self.opponent_agent(self.boards[1].observation, self.env.configuration)
            next_kore_action.append(next_opponent_action)

            self.opponent_action_names.append('Balanced Agent')  # w&b login purposes

        else:  # balanced agent
            self.opponent_agent = self.env.agents[self.enemy_agent]
            next_opponent_action = self.opponent_agent(self.boards[1].observation, self.env.configuration)
            next_kore_action.append(next_opponent_action)

            self.opponent_action_names.append('Balanced Agent')  # w&b login purposes

        # print("next_opponent_action: ", next_opponent_action)

        step_result = self.env.step(
            next_kore_action)  # Env step NOT our step function defined above. This is the kore action.
        self.boards = get_boards_from_kore_env_state(step_result, self.env.configuration)

        self.shipyard_actions = {}
        self.shipyard_action_names = {}

        self.shipyards = self.boards[0].current_player.shipyards
        self.num_shipyards = len(self.shipyards)

        self.shipyards_enemy = self.boards[1].current_player.shipyards
        self.num_shipyards_enemy = len(self.shipyards_enemy)

        self.last_game_step_board = copy.deepcopy(self.boards[0])
        self.last_game_step_board_enemy = copy.deepcopy(self.boards[1])

    def _shipyard_sub_step(self):
        """
        - Simulates all accumulated actions on the board and updates the board in-place
        """
        # probably too much copying since next() also makes a shallow copy
        # but better safe than sorry :)
        current_copy = copy.deepcopy(self.last_game_step_board)
        current_copy_enemy = copy.deepcopy(self.last_game_step_board_enemy)

        for shipyard in current_copy.current_player.shipyards:
            if shipyard.id in self.shipyard_actions:  # if there is an action for a shipyard
                action_object = ShipyardAction.from_str(self.shipyard_actions[shipyard.id])
                shipyard.next_action = action_object

        for shipyard in current_copy_enemy.current_player.shipyards:
            if shipyard.id in self.shipyard_actions:  # if there is an action for a shipyard
                action_object = ShipyardAction.from_str(self.shipyard_actions[shipyard.id])
                shipyard.next_action = action_object

        self.boards[self.player_id] = current_copy.next()
        self.boards[self.enemy_id] = current_copy_enemy.next()

    def setup_enemy(self, p=0.8, q=0.3):
        """Load a previously saved model as the enemy.
        It loads the last saved model with probability p or loads a random model otherwise.
        With probability q, we sample balanced agent"""

        self.enemy_models_dict = self._load_all_models()

        self.enemy_models = self.enemy_models_dict['model']
        self.enemy_model_names = self.enemy_models_dict['model name']

        k = random.uniform(0, 1)

        if (not self.enemy_models) or (len(self.enemy_models) < 3) or (
                k < q):  # checking if there is no preexisting model in the model folder
            # or if there are not enough model saved  # 6
            # or sample the balanced agent with probability q
            # print("Using balanced agent")
            self.opponent_agent = self.env.agents[self.enemy_agent]  # use balanced agent
            self.opponent_agent_name = 0

        else:  # pick an agent from the folder using 80/20 rule
            j = random.uniform(0, 1)
            if j < p:  # pick the last (best) agent
                self.opponent_agent = self.enemy_models[-1]
                self.opponent_agent_name = self.enemy_model_names[-1]

            else:  # select a random agent
                start = 0
                end = len(self.enemy_models) - 1
                i = random.randint(start, end)
                self.opponent_agent = self.enemy_models[i]
                self.opponent_agent_name = self.enemy_model_names[i]


    def _load_model(self, name):
        model_path = os.path.join(os.path.abspath("../selfplay_models"), self.model_name, self.run_id, name)
        # print("model path: ", model_path)
        if os.path.exists(model_path):
            # print(f'Loading {name}')

            try:
                ppo_model = PPO.load(model_path)
                # print("loaded PPO successfully")
                return ppo_model
            except Exception as e:
                time.sleep(5)
                print(e)
        else:
            raise Exception(f'\n{model_path} not found')

    def _load_all_models(self):
        path = os.path.join(os.path.abspath("../selfplay_models/"), self.model_name, self.run_id)
        model_list = [f for f in os.listdir(path) if f.startswith("selfplay")]
        model_list.sort()
        models = {'model': [],
                  'model name': []}
        for model_name in model_list[-self.self_play_window:]:
            models['model'].append(self._load_model(name=model_name))
            models['model name'].append(self._model_name_int(model_name))
        return models

    def _model_name_int(self, model_name_str):
        return int(model_name_str.split("_")[1])
