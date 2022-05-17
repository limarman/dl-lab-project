import numpy as np
from kaggle_environments.envs.kore_fleets.helpers import Board

from src.States.KoreState import KoreState
from src.States.StateAdapter import StateAdapter


class DummyAdapter(StateAdapter):

    def board_to_state(self, board: Board) -> KoreState:
        return KoreState(values=np.array(
            [
                board.current_player.kore,
                board.current_player.shipyards[0].ship_count,
                board.current_player.shipyards[0].max_spawn,
             ]
            + board.observation["kore"]
        ))

