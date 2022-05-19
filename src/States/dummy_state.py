import numpy as np

from kaggle_environments.envs.kore_fleets.helpers import *

from src.States.board_wrapper import BoardWrapper
from src.States.kore_state import KoreState

from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board


class DummyState(KoreState):

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
        tensor = self._get_tensor()
        super(DummyState, self).__init__(tensor.shape, tensor, self.board_wrapper)

    def _get_tensor(self):
        """
        Puts all state values in a torch tensor, eg. as input for a simple MLP
        :return: np array of size (21*21 + 3*1)
        """
        data = self.kore_map + [
                self.kore_me,
                self.ship_count_me,
                self.max_spawn
            ]

        return np.array(data)

    @staticmethod
    def get_input_shape():
        dummy_board = create_board()
        return DummyState(dummy_board).tensor.shape[0]
