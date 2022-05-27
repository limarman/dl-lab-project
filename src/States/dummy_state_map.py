from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board
from kaggle_environments.envs.kore_fleets.helpers import Board

from src.States.board_wrapper import BoardWrapper
import numpy as np

from src.States.kore_state import KoreState


class DummyStateMap(KoreState):

    def __init__(self, board: Board):
        """
        Initializes all relevant state values via the BoardWrapper
        :param board: board as provided by the Kaggle env
        """
        self.board_wrapper = BoardWrapper(board)
        self.kore_map = self.board_wrapper.get_kore_map()
        self.kore_me = self.board_wrapper.get_kore_me()
        self.max_spawn = self.board_wrapper.get_max_spawn_me()
        self.ship_count_me = self.board_wrapper.get_ship_count_me()
        self.shipyard_fleets_map = self.board_wrapper.get_shipyard_fleets_map()
        self.shipyard_count_me = len(self.board_wrapper.get_shipyards_of_current_player())
        tensor = self._get_tensor()
        super(DummyStateMap, self).__init__(tensor.shape, tensor, self.board_wrapper)

    @staticmethod
    def get_input_shape():
        dummy_board = create_board()
        return DummyStateMap(dummy_board).tensor.shape

    def _get_tensor(self):
        kore_map = np.array(self.kore_map).arrange(21,21)
        shipyards_fleets_map = self.shipyard_fleets_map

        return np.stack([kore_map, shipyards_fleets_map])

