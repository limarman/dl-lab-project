import math

from typing import Dict

from src.States.board_wrapper import BoardWrapper


class ActionAdapter:

    N_ACTIONS: int = 100
    MANEUVER_LOOKUP: Dict[int, str] = {
        0: "S",
        1: "N4",
        2: "E4",
        3: "S4",
        4: "W4",
        5: "N2E5S2W",
        6: "N2W5S2E",
        7: "S2E5N2W",
        8: "S2W5N2E",
        9: "S6E6N6W",
    }

    def __init__(self):
        pass

    def agent_to_kore_action(self, agent_action: int, board_wrapper: BoardWrapper) -> Dict[str, str]:
        agent_action_str = str(agent_action)

        if len(agent_action_str) < 2:
            agent_action_str = agent_action_str.zfill(2)

        shipyard_idx = int(agent_action_str[0])
        flight_plan_idx = int(agent_action_str[1])

        shipyard_id = self.__select_shipyard_id(board_wrapper, shipyard_idx)
        base_action = self.__select_base_action(flight_plan_idx)

        if base_action == "SPAWN":
            ship_count = str(board_wrapper.get_max_spawn_shipyard(shipyard_idx))
            kore_action = f"{base_action}_{ship_count}"
        else:
            ship_count = board_wrapper.get_ship_count_shipyard(shipyard_idx)
            if ship_count > 0:
                direction = self.__select_maneuver(int(ship_count), flight_plan_idx)
                build_indicator = self.__select_build_indicator(ship_count)

                kore_action = f"{base_action}_{ship_count}{direction}{build_indicator}"
            else:
                kore_action = ""

        kore_action = {shipyard_id: kore_action}

        return kore_action

    def __select_shipyard_id(self, board_wrapper: BoardWrapper, shipyard_idx: int) -> str:
        shipyards = board_wrapper.get_shipyards_of_current_player()
        if len(shipyards) == 0:
            return ""
        if shipyard_idx >= len(shipyards):
            shipyard_idx = len(shipyards) - 1
        return board_wrapper.get_shipyards_of_current_player()[shipyard_idx].id

    def __select_base_action(self, flight_plan_idx: int) -> str:
        return "SPAWN" if flight_plan_idx == 0 else "LAUNCH"

    def __select_build_indicator(self, ship_count: int):
        if ship_count >= 50:
            return "C"
        return ""

    def __select_maneuver(self, ship_count: int, flight_plan_idx):
        max_maneuver_length = math.floor(2 * math.log(ship_count)) + 1
        maneuver = self.MANEUVER_LOOKUP[flight_plan_idx]

        if len(maneuver) > max_maneuver_length:
            flight_plan_idx = 0

        return f"_{self.MANEUVER_LOOKUP[flight_plan_idx]}"

