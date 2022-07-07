from typing import Dict

from kaggle_environments.envs.kore_fleets.helpers import Board
from src.States.kore_state import KoreState


def get_boards_from_kore_env_state(state_dict, config):
    """
    Adapted from https://www.kaggle.com/code/obkyrush/kore-env-multi-agent-interaction/notebook

    Transfers the raw observations from the kaggle environments to boards for each player
    """
    # only the first observation contains all relevant observations
    # the other observations only contains the player-specific observation
    common_obs = {common_key: state_dict[0]['observation'][common_key] for common_key in ('players', 'step', 'kore')}

    player_boards = {}

    for player_state in state_dict:
        player_state['observation'].update(common_obs)
        board = Board(player_state['observation'], config)
        player_id = player_state['observation'].player
        player_boards[player_id] = board

    return player_boards


def get_info_logs(state: KoreState, actions: Dict[str, str], action_names: Dict[str, str]):
    """ Calculates some basic metrics for wandb logger"""
    none_actions = [1 if (action == 'None' or not action) else 0 for action in actions.values()]
    # for action in actions:
    #     if actions[action] == "None":
    #         print(f"invalid action: {action_names[action]}")

    info = {
        'none_actions': none_actions,
        'game_length': state.board_wrapper.board.step,
        'kore_me': state.kore_me,
        'kore_delta': state.kore_me - state.board_wrapper.get_kore_opponent(),
        'shipyard_count_me': state.board_wrapper.get_shipyard_count_me(),
        'shipyard_count_opponent': state.board_wrapper.get_shipyard_count_opponent(),
        'fleet_count_me': state.board_wrapper.get_fleet_count_me(),
        'fleet_count_opponent': state.board_wrapper.get_fleet_count_opponent(),
        'actions': [name for name in action_names.values()]
    }

    return info
