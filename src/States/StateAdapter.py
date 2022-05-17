from abc import ABC, abstractmethod

from kaggle_environments.envs.kore_fleets.helpers import Board

from src.States.KoreState import KoreState


class StateAdapter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def board_to_state(self, board: Board) -> KoreState:
        pass
