from kaggle_environments.envs.kore_fleets.helpers import Board


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

