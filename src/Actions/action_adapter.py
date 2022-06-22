from abc import abstractmethod, ABC
from typing import Dict

from numpy import ndarray

from src.States.board_wrapper import BoardWrapper


class ActionAdapter(ABC):

    N_ACTIONS = 0

    def __init__(self):
        pass

    @abstractmethod
    def agent_to_kore_action(self, agent_action: ndarray, board_wrapper: BoardWrapper) -> (Dict[str, str], Dict[str, str]):
        """
        Returns the Kore action (first Dict) and the high-level action name for logging purposes (second dict)
        """
        pass
