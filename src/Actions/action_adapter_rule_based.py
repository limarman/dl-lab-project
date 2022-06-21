import itertools

import numpy as np
from numpy import ndarray

from src.Actions.action_adapter import ActionAdapter
from src.Actions.rule_based_actor import RuleBasedActor

from typing import Dict

from src.States.board_wrapper import BoardWrapper
from kaggle_environments.envs.kore_fleets.helpers import *


class RuleBasedActionAdapter(ActionAdapter):

    N_ACTIONS: int = 5 #expand, attack, box-farm, axis-farm, build

    def __init__(self, single_shipyard=False):
        super().__init__()
        if single_shipyard:
            self.N_ACTIONS = 4

    def agent_to_kore_action(self, agent_actions: ndarray, board_wrapper: BoardWrapper) -> Dict[str, str]:
        #shipyard_idx = agent_action % 10
        player_shipyards = board_wrapper.player_me.shipyards
        kore_action = {}

        if not player_shipyards:
            return {}

        for shipyard in player_shipyards:
            invalid_actions_mask = self.get_invalid_action_mask(shipyard, board_wrapper.board)
            valid_agent_actions = [agent_actions[i] * invalid_actions_mask[i] for i in range(len(agent_actions))]
            valid_agent_actions = np.asarray(valid_agent_actions)
            if np.sum(valid_agent_actions) != 0:
                valid_agent_actions_probs = valid_agent_actions / np.sum(valid_agent_actions)
                action_idx = np.random.choice(a=range(len(agent_actions)), p=valid_agent_actions_probs)
            else:
                action_idx = np.random.choice(a=range(len(agent_actions)))

            action = str(self._get_rb_action(action_idx, shipyard, board_wrapper.board))
            kore_action[shipyard.id] = action

        return kore_action

    def get_invalid_action_mask(self, shipyard: Shipyard, board: Board):
        rba = RuleBasedActor(board)
        possible_actions = [
            rba.build_max(shipyard),
            rba.start_optimal_axis_farmer(shipyard, 9),
            rba.start_optimal_box_farmer(shipyard, 9),
            rba.attack_closest(shipyard),
            rba.expand_optimal(shipyard),
        ]

        invalid_actions_mask = [
            0 if action is None
            else 1
            for action in possible_actions
        ]

        return invalid_actions_mask

    @staticmethod
    def _get_rb_action(action_idx, shipyard: Shipyard, board: Board):
        rba = RuleBasedActor(board)
        if action_idx == 0:
            shipyard_action = rba.build_max(shipyard)
            # print("build")
        elif action_idx == 1:
            shipyard_action = rba.start_optimal_axis_farmer(shipyard, 9)
        elif action_idx == 2:
            # print("farmer")
            shipyard_action = rba.start_optimal_box_farmer(shipyard, 9)
        elif action_idx == 3:
            # print("attack")
            shipyard_action = rba.attack_closest(shipyard)
        elif action_idx == 4:
            # print("expand")
            shipyard_action = rba.expand_optimal(shipyard)
        else:
            shipyard_action = None

        return shipyard_action


    @staticmethod
    def __select_shipyard(board_wrapper: BoardWrapper, shipyard_idx: int) -> Shipyard:
        shipyards = board_wrapper.get_shipyards_of_current_player()
        if len(shipyards) == 0:
            return None
        if shipyard_idx >= len(shipyards):
            shipyard_idx = len(shipyards) - 1
        return board_wrapper.get_shipyards_of_current_player()[shipyard_idx]


