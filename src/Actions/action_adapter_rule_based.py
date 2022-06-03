import itertools

from src.Actions.action_adapter import ActionAdapter
from src.Actions.rule_based_actor import RuleBasedActor

from typing import Dict

from src.States.board_wrapper import BoardWrapper
from kaggle_environments.envs.kore_fleets.helpers import *


class ActionAdapterRuleBased(ActionAdapter):

    N_ACTIONS: int = 5 #expand, attack, box-farm, axis-farm, build

    def __init__(self, single_shipyard=False):
        super().__init__()
        if single_shipyard:
            self.N_ACTIONS = 4

    def agent_to_kore_action(self, agent_action: int, board_wrapper: BoardWrapper) -> Dict[str, str]:
        #shipyard_idx = agent_action % 10
        action_idx = agent_action

        shipyard = self.__select_shipyard(board_wrapper, 0)

        if shipyard is None:
            return {}

        shipyard_action = self._get_rb_action(action_idx, shipyard, board_wrapper.board)

        kore_action = {shipyard.id: str(shipyard_action)}

        return kore_action

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
            # why not raise exception here?
            # raise Exception('Action index is out of bounds')
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


