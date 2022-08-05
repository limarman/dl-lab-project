from abc import ABC, abstractmethod

from src.States.kore_state import KoreState, Dict


class KoreReward(ABC):
    """
    Simple interface for reward implementations
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_reward_from_action(self, state: KoreState, actions) -> float:
        """
        Calculates scalar reward value
        """
        pass

    @abstractmethod
    def get_reward_from_states(self, previous_state: KoreState, next_state: KoreState):
        pass

    @abstractmethod
    def get_reward(self, previous_state: KoreState, next_state: KoreState, action: Dict[str, str]):
        pass

    def reset(self):
        pass
