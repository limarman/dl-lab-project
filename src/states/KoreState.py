from abc import ABC

from src.states import BoardWrapper


class KoreState(ABC):
    """
    Simple abstract class for standardizing most important state features
    """

    def __init__(self, input_shape, tensor, board_wrapper: BoardWrapper):
        """

        :param input_shape: shape of the input tensor
        :param tensor: input tensor of state
        :param board:
        """
        self.input_shape = input_shape
        self.tensor = tensor
        self.board_wrapper = board_wrapper
