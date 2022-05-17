from src.Rewards.KoreReward import KoreReward
from src.States.DummyState import DummyState


class DummyReward(KoreReward):
    """
    Dummy reward implementation for quick prototyping
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """
        pass

    @staticmethod
    def get_reward_from_action(self, current_state: DummyState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = current_state.apply_action_to_board(actions)
        return self.get_reward_from_states(self.current_state, next_state)

    def get_reward_from_states(self, previous_state, next_state):
        kore_delta = previous_state.kore_me - next_state.kore_me

        return max(kore_delta, 0)
