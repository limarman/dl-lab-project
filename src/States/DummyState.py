import numpy as np

from src.States.KoreState import KoreState
from src.States.BoardWrapper import *


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
        :return: floatTensor of size (3*21*21 + 5*1)
        TODO adjust shapes
        """
        data = self.kore_map + [
                self.kore_me,
                self.ship_count_me,
                self.max_spawn
            ]

        return np.array(data)
