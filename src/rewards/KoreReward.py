from abc import ABC, abstractmethod


class KoreReward(ABC):
    """
    Simple interface for reward implementations
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_reward(self, state, action) -> float:
        """
        Calculates scalar reward value
        """
        pass
