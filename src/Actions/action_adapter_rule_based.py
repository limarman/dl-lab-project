from src.Actions.action_adapter import ActionAdapter
from src.Actions.rule_based_actor import RuleBasedActor

from typing import Dict

from src.States.board_wrapper import BoardWrapper
from kaggle_environments.envs.kore_fleets.helpers import Shipyard


class ActionAdapterRuleBased(ActionAdapter):

    N_ACTIONS: int = 40 #expand, attack, farm, build

    def __init__(self):
        super().__init__()

    def agent_to_kore_action(self, agent_action: int, board_wrapper: BoardWrapper) -> Dict[str, str]:
        shipyard_idx = agent_action % 10
        action_idx = int(agent_action / 10)

        shipyard = self.__select_shipyard(board_wrapper, shipyard_idx)

        if shipyard is None:
            return {}

        rba = RuleBasedActor(board_wrapper.board)
        shipyard_action = None

        if action_idx == 0:
            shipyard_action = rba.build_max(shipyard)
            # print("build")
        elif action_idx == 1:
            # print("farmer")
            shipyard_action = rba.start_optimal_box_farmer(shipyard, 9)
        elif action_idx == 2:
            # print("expand")
            shipyard_action = rba.expand_randomly(shipyard, 4, 1)
        elif action_idx == 3:
            # print("attack")
            shipyard_action = rba.attack_closest(shipyard)

        kore_action = {shipyard.id: str(shipyard_action)}

        return kore_action

    @staticmethod
    def __select_shipyard(board_wrapper: BoardWrapper, shipyard_idx: int) -> Shipyard:
        shipyards = board_wrapper.get_shipyards_of_current_player()
        if len(shipyards) == 0:
            return None
        if shipyard_idx >= len(shipyards):
            shipyard_idx = len(shipyards) - 1
        return board_wrapper.get_shipyards_of_current_player()[shipyard_idx]


