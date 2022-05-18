from abc import ABC
from typing import List

from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from kaggle_environments.envs.kore_fleets.helpers import *

from src.States import board_wrapper


class KoreState(ABC):
    """
    Simple abstract class for standardizing most important state features
    """

    def __init__(self, input_shape, tensor, board_wrapper: board_wrapper):
        """
        :param input_shape: shape of the input tensor
        :param tensor: input tensor of state
        :param board:
        """
        self.input_shape = input_shape
        self.tensor = tensor
        self.board_wrapper = board_wrapper

    def apply_action_to_board(self, actions: List[ShipyardAction]):
        """
        Applies the shipyard actions to the current state of the current player

        :param state: current state
        :param actions: actions to be applied to the shipyards
        :return: new state after applying the actions
        """
        for shipyard, action in zip(self.board_wrapper.board.current_player.shipyards, actions):
            shipyard.next_action = action
        next_board = self.board_wrapper.board.next()
        # calling constructur of subclass
        next_state = type(self)(next_board)

        return next_state
