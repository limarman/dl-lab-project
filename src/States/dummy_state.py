import numpy as np

from kaggle_environments.envs.kore_fleets.helpers import *

from src.States.board_wrapper import BoardWrapper
from src.States.kore_state import KoreState


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
        self.shipyard_map = self.board_wrapper.get_shipyard_map()
        self.shipyard_count_me = len(self.board_wrapper.get_shipyards_of_current_player())
        tensor = self._get_tensor()
        super(DummyState, self).__init__(tensor.shape, tensor, self.board_wrapper)

    def _get_tensor(self):
        """
        Puts all state values in a torch tensor, eg. as input for a simple MLP
        :return: np array of size (21*21 + 3*1)
        """
        list_data = self.kore_map + [
                self.kore_me,
                self.ship_count_me,
                self.max_spawn,
                self.shipyard_count_me,
            ]

        print(self.shipyard_map)
        data = np.concatenate((np.array(list_data), self.shipyard_map))

        return np.array(data)
