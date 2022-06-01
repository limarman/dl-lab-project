import numpy as np
from kaggle_environments.envs.kore_fleets.helpers import *

from src.States.board_wrapper import BoardWrapper
from src.States.kore_state import KoreState

from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board


class AdvancedState(KoreState):

    def __init__(self, board: Board):
        """
        Initializes all relevant state values via the BoardWrapper
        :param board: board as provided by the Kaggle env
        """
        self.board_wrapper = BoardWrapper(board)
        self.kore_map = self.board_wrapper.get_kore_map()
        self.kore_me = self.board_wrapper.get_kore_me()
        self.kore_opponent = self.board_wrapper.get_kore_opponent()
        self.ship_count_me = self.board_wrapper.get_ship_count_me()
        self.ship_count_opponent = self.board_wrapper.get_ship_count_opponent()
        self.fleet_pos = self.board_wrapper.get_fleet_pos()
        self.shipyards_pos = self.board_wrapper.get_shipyard_pos()
        self.shipyard_count_me = self.board_wrapper.get_shipyard_count_me()
        self.shipyard_count_opponent = self.board_wrapper.get_shipyard_count_opponent()
        self.lost = np.all(self.shipyards_pos <= 0)
        self.step_normalized = self.board_wrapper.get_step()
        self.max_spawn_me = self.board_wrapper.get_max_spawn_me()
        self.cargo = self.board_wrapper.get_cargo()
        tensor = self._get_tensor()
        super(AdvancedState, self).__init__(tensor.shape, tensor, self.board_wrapper)

    def _get_tensor(self):
        """
        Puts all state values in a torch tensor, eg. as input for a simple MLP
        :return: floatTensor of size (3*21*21 + 6*1)
        """
        data_list = ([
            [self.kore_me],
            [self.kore_opponent],
            [self.ship_count_me],
            [self.ship_count_opponent],
            [self.step_normalized],
            [self.max_spawn_me],
            [self.shipyard_count_me],
            [self.shipyard_count_opponent],
            self.kore_map,
            self.fleet_pos,
            self.shipyards_pos
        ])

        tensor = np.concatenate([np.array(data).flatten() for data in data_list]).squeeze()

        return tensor

    @staticmethod
    def get_input_shape() -> int:
        dummy_board = create_board()
        return AdvancedState(dummy_board).input_shape[0]




