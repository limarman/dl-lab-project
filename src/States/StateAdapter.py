from abc import ABC, abstractmethod

from src.States.KoreState import KoreState


class StateAdapter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def to_state(self, obs, config) -> KoreState:
        pass
