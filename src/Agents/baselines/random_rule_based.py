from kaggle_environments.envs.kore_fleets.helpers import *

from src.Actions.rule_based_actor import RuleBasedActor

"""
Simple agent that takes one of our high-level-actions uniformly at random
Intended to verify that we are learning something
"""


def agent_without_expand(obs, config):
    return _agent(obs, config, True)

def agent_with_expand(obs, config):
    return _agent(obs, config, False)


def _agent(obs, config, single_shipyard: bool):
    board = Board(obs, config)
    me = board.current_player
    rba = RuleBasedActor(board)

    upper_bound = 3 if single_shipyard else 4

    for shipyard in me.shipyards:
        action_idx = random.randint(0, upper_bound)

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
            shipyard_action = rba.expand_right(shipyard)
        else:
            shipyard_action = None

        shipyard.next_action = shipyard_action

    return me.next_actions
