from abc import abstractmethod, ABC
from typing import Dict

from src.States.board_wrapper import BoardWrapper


class ActionAdapter(ABC):

    N_ACTIONS = 0

    def __init__(self):
        pass

    @abstractmethod
    def agent_to_kore_action(self, agent_action: int, board_wrapper: BoardWrapper) -> Dict[str, str]:
        pass
