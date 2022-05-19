from numpy import ndarray
from typing import Dict


class ActionAdapter:

    N_ACTIONS = 5

    def __init__(self):
        pass

    def agent_to_kore_action(self, agent_action: int) -> Dict[str, str]:
        kore_action = {}
        if agent_action == 0:
            kore_action["0-1"] = "SPAWN_1"
        elif agent_action == 1:
            kore_action["0-1"] = "LAUNCH_2_N"
        elif agent_action == 2:
            kore_action["0-1"] = "LAUNCH_2_W"
        elif agent_action == 3:
            kore_action["0-1"] = "LAUNCH_50_W2C"
        elif agent_action == 4:
            kore_action["0-1"] = "LAUNCH_50_N2C"

        return kore_action
