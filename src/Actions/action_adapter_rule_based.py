import numpy as np
from numpy import ndarray

from src.Actions.action_adapter import ActionAdapter
from src.Actions.action_validity_masking import get_action_validity_mask
from src.Actions.rule_based_actor import RuleBasedActor

from src.States.board_wrapper import BoardWrapper
from kaggle_environments.envs.kore_fleets.helpers import *


class ActionAdapterRuleBased(ActionAdapter):

    N_ACTIONS: int = 7 #expand, attack, box-farm, axis-farm, build, wait

    def __init__(self, single_shipyard=False):
        super().__init__()
        if single_shipyard:
            self.N_ACTIONS = 4

    def agent_to_kore_action(self, agent_actions: ndarray, board_wrapper: BoardWrapper, shipyard) -> (Dict[str, str], Dict[str, str]):
        valid_action_mask = get_action_validity_mask(shipyard, board_wrapper.board)
        valid_agent_actions = [agent_actions[i] * valid_action_mask[i] for i in range(len(agent_actions))]
        valid_agent_actions = np.asarray(valid_agent_actions)
        if np.sum(valid_agent_actions) != 0:
            valid_agent_actions_probs = valid_agent_actions / np.sum(valid_agent_actions)
            action_idx = np.random.choice(a=range(len(agent_actions)), p=valid_agent_actions_probs)
        else:
            # take a random valid action if possible
            choice_actions = [i for i, val in enumerate(valid_action_mask) if val == 1]
            if not choice_actions:
                choice_actions = range(len(agent_actions))
            action_idx = np.random.choice(a=choice_actions)

        action, name = self._get_rb_action(action_idx, shipyard, board_wrapper.board)
        kore_action = {shipyard.id: str(action)}
        kore_action_names = {shipyard.id: name}

        return kore_action, kore_action_names

    @staticmethod
    def _get_rb_action(action_idx, shipyard: Shipyard, board: Board):
        rba = RuleBasedActor(board)
        if action_idx == 0:
            shipyard_action = rba.build_max(shipyard)
            action_name = "Build"
        elif action_idx == 1:
            shipyard_action = rba.start_optimal_box_farmer(shipyard, 9)
            action_name = "Box Farming"
        elif action_idx == 2:
            shipyard_action = rba.start_optimal_axis_farmer(shipyard, 9)
            action_name = "Axis Farming"
        elif action_idx == 3:
            shipyard_action = rba.attack_closest(shipyard)
            action_name = "Attack"
        elif action_idx == 4:
            shipyard_action = rba.expand_right(shipyard)
            action_name = "Expand_Right"
        elif action_idx == 5:
            shipyard_action = rba.expand_down(shipyard)
            action_name = "Expand_Down"
        elif action_idx == 6:
            shipyard_action = rba.wait(shipyard)
            action_name = "Wait"
        else:
            shipyard_action = None
            action_name = None

        return shipyard_action, action_name


    @staticmethod
    def __select_shipyard(board_wrapper: BoardWrapper, shipyard_idx: int) -> Shipyard:
        shipyards = board_wrapper.get_shipyards_of_current_player()
        if len(shipyards) == 0:
            return None
        if shipyard_idx >= len(shipyards):
            shipyard_idx = len(shipyards) - 1
        return board_wrapper.get_shipyards_of_current_player()[shipyard_idx]


