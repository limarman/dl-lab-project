import numpy
import numpy as np

from kaggle_environments.envs.kore_fleets.helpers import *

from src.States.board_wrapper import BoardWrapper
from src.States.kore_state import KoreState

from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board


class MapState(KoreState):

    def __init__(self, board: Board):
        """
        Initializes all relevant state values via the BoardWrapper
        :param board: board as provided by the Kaggle env
        """
        self.board_wrapper = BoardWrapper(board)
        self.kore_map = self.board_wrapper.get_overlapping_kore_map()
        self.ship_map = self.board_wrapper.get_ship_map()
        self.max_spawn_map = self.board_wrapper.get_max_spawn_map()
        self.kore_me = self.board_wrapper.get_kore_me()
        self.kore_opponent = self.board_wrapper.get_kore_opponent()
        self.step = self.board_wrapper.get_step(True)

        #TODO: Current Kore State would not allow concatenating architectures
        #TODO: Include scalar information into the state
        tensor = self._get_tensor()
        super(MapState, self).__init__(tensor.shape, tensor, self.board_wrapper)

    def _get_tensor(self):
        """
        """
        kore_map = self.kore_map
        ship_map = self.ship_map
        max_spawn_map = self.max_spawn_map

        return np.stack([kore_map, ship_map, max_spawn_map])

    @staticmethod
    def get_input_shape():
        dummy_board = create_board()
        return MapState(dummy_board)._get_tensor().shape
