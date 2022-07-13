from typing import List

from kaggle_environments.envs.kore_fleets.helpers import Shipyard, Board

from src.Actions.rule_based_actor import RuleBasedActor


def get_action_validity_mask(shipyard: Shipyard, board: Board) -> List[int]:
    rba = RuleBasedActor(board)
    possible_actions = [
        rba.build_max(shipyard, validity_check=True),
        rba.start_optimal_box_farmer(shipyard, 9, validity_check=True),
        rba.start_optimal_axis_farmer(shipyard, 9, validity_check=True),
        rba.attack_closest(shipyard, validity_check=True),
        rba.expand_right(shipyard, validity_check=True),
        rba.expand_down(shipyard, validity_check=True),
        rba.wait(shipyard, validity_check=True),
    ]

    valid_action_mask = [
        0 if action is None
        else 1
        for action in possible_actions
    ]

    return valid_action_mask
