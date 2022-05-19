from kaggle_environments.envs.kore_fleets.helpers import Board
from numpy import ndarray
from typing import Dict, List

from src.States.board_wrapper import BoardWrapper


class ActionAdapter:

    N_ACTIONS: int = 1000
    FLEETS_AMOUNT_LOOKUP_LIST: List[str] = ["1", "2", "5", "10", "15", "20", "30", "40", "50", "100"]
    DIRECTION_LOOKUP_LIST: List[str] = ["N", "E", "S", "W"]
    N_MOVES_IN_DIRECTION: int = 4

    def __init__(self):
        pass

    def agent_to_kore_action(self, agent_action: int, board_wrapper: BoardWrapper) -> Dict[str, str]:
        agent_action_str = str(agent_action)

        if len(agent_action_str) < 3:
            agent_action_str = agent_action_str.zfill(3)

        shipyard_idx = int(agent_action_str[0])
        fleets_amount_idx = int(agent_action_str[1])
        flight_plan_idx = int(agent_action_str[2])

        shipyard_id = self.__select_shipyard_id(board_wrapper, shipyard_idx)
        base_action = self.__select_base_action(flight_plan_idx)
        fleets_amount = self.__select_fleets_amount(fleets_amount_idx)
        direction = self.__select_direction(flight_plan_idx)
        build_indicator = self.__select_build_indicator(flight_plan_idx)

        kore_action = f"{base_action}{fleets_amount}{direction}{build_indicator}"

        kore_action = {shipyard_id: kore_action}

        return kore_action

    def __select_shipyard_id(self, board_wrapper: BoardWrapper, shipyard_idx: int) -> str:
        shipyards = board_wrapper.get_shipyards_of_current_player()
        if shipyard_idx >= len(shipyards):
            return ""
        return board_wrapper.get_shipyards_of_current_player()[shipyard_idx].id

    def __select_base_action(self, flight_plan_idx: int) -> str:
        return "SPAWN" if flight_plan_idx in [0, 1] else "LAUNCH"

    def __select_fleets_amount(self, fleets_amount_idx):
        return f"_{self.FLEETS_AMOUNT_LOOKUP_LIST[fleets_amount_idx]}"

    def __select_build_indicator(self, flight_plan_idx):
        if flight_plan_idx % 2 == 1 and flight_plan_idx > 1:
            return "C"
        return ""

    def __select_direction(self, flight_plan_idx):
        if flight_plan_idx in [0, 1]:
            return ""
        return f"_{self.DIRECTION_LOOKUP_LIST[flight_plan_idx % 4]}{str(self.N_MOVES_IN_DIRECTION)}"

