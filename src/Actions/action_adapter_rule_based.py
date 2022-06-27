import numpy as np
from numpy import ndarray

from src.Actions.action_adapter import ActionAdapter
from src.Actions.rule_based_actor import RuleBasedActor

from src.States.board_wrapper import BoardWrapper
from kaggle_environments.envs.kore_fleets.helpers import *


class ActionAdapterRuleBased(ActionAdapter):

    N_ACTIONS: int = 5 #expand, attack, box-farm, axis-farm, build

    def __init__(self, single_shipyard=False):
        super().__init__()
        if single_shipyard:
            self.N_ACTIONS = 4

    def agent_to_kore_action(self, agent_actions: ndarray, board_wrapper: BoardWrapper) -> (Dict[str, str], Dict[str, str]):
        #shipyard_idx = agent_action % 10
        player_shipyards = board_wrapper.player_me.shipyards
        kore_action = {}
        kore_action_names = {}

        if not player_shipyards:
            return {}, {}

        for shipyard in player_shipyards:
            invalid_actions_mask = self.get_invalid_action_mask(shipyard, board_wrapper.board)
            valid_agent_actions = [agent_actions[i] * invalid_actions_mask[i] for i in range(len(agent_actions))]
            valid_agent_actions = np.asarray(valid_agent_actions)
            if np.sum(valid_agent_actions) != 0:
                valid_agent_actions_probs = valid_agent_actions / np.sum(valid_agent_actions)
                action_idx = np.random.choice(a=range(len(agent_actions)), p=valid_agent_actions_probs)
            else:
                # take a random valid action if possible
                choice_actions = [i for i, val in enumerate(invalid_actions_mask) if val == 1]
                if not choice_actions:
                    choice_actions = range(len(agent_actions))
                action_idx = np.random.choice(a=choice_actions)


            action, action_name = self._get_rb_action(action_idx, shipyard, board_wrapper.board)
            kore_action[shipyard.id] = str(action)
            kore_action_names[shipyard.id] = action_name

        return kore_action, kore_action_names

    def get_invalid_action_mask(self, shipyard: Shipyard, board: Board):
        rba = RuleBasedActor(board)
        possible_actions = [
            rba.build_max(shipyard, validity_check=True),
            rba.start_optimal_axis_farmer(shipyard, 9, validity_check=True),
            rba.start_optimal_box_farmer(shipyard, 9, validity_check=True),
            rba.attack_closest(shipyard, validity_check=True),
            rba.expand_optimal(shipyard, validity_check=True),
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
            action_name = "Build"
        elif action_idx == 1:
            shipyard_action = rba.start_optimal_axis_farmer(shipyard, 9)
            action_name = "Axis Farming"
        elif action_idx == 2:
            shipyard_action = rba.start_optimal_box_farmer(shipyard, 9)
            action_name = "Box Farming"
        elif action_idx == 3:
            shipyard_action = rba.attack_closest(shipyard)
            action_name = "Attack"
        elif action_idx == 4:
            shipyard_action = rba.expand_optimal(shipyard)
            action_name = "Expand"
        else:
            shipyard_action = None

        return shipyard_action, action_name


    @staticmethod
    def __select_shipyard(board_wrapper: BoardWrapper, shipyard_idx: int) -> Shipyard:
        shipyards = board_wrapper.get_shipyards_of_current_player()
        if len(shipyards) == 0:
            return None
        if shipyard_idx >= len(shipyards):
            shipyard_idx = len(shipyards) - 1
        return board_wrapper.get_shipyards_of_current_player()[shipyard_idx]


