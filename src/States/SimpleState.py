import torch
import numpy as np
from kaggle_environments.envs.halite.helpers import Board

from src.States.BoardWrapper import BoardWrapper
from src.States.KoreState import KoreState


class SimpleState(KoreState):

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
        # TODO replace by own function and add test
        self.lost = np.all(self.shipyards_pos <= 0)
        self.step_normalized = self.board_wrapper.get_step()
        # TODO add max_spawn
        tensor = self._get_tensor()
        super(SimpleState, self).__init__(tensor.shape, tensor, self.board_wrapper)

    def _get_tensor(self):
        """
        Puts all state values in a torch tensor, eg. as input for a simple MLP
        :return: floatTensor of size (3*21*21 + 5*1)
        """
        data_list = ([
            [self.kore_me],
            [self.kore_opponent],
            [self.ship_count_me],
            [self.ship_count_opponent],
            [self.step_normalized],
            self.kore_map,
            self.fleet_pos,
            self.shipyards_pos
        ])

        tensor = np.concatenate([np.array(data).flatten() for data in data_list])

        return tensor




