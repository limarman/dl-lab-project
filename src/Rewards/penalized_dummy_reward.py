from src.Rewards.kore_reward import KoreReward
from src.States.dummy_state import DummyState


class PenalizedDummyReward(KoreReward):
    """
    Dummy reward implementation for quick prototyping
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """
        pass

    @staticmethod
    def get_reward_from_action(current_state: DummyState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = current_state.apply_action_to_board(actions)
        return PenalizedDummyReward.get_reward_from_states(current_state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: DummyState, next_state: DummyState):
        kore_delta = next_state.kore_me - previous_state.kore_me

        return max(kore_delta, 0)

    @staticmethod
    def get_reward(self, previous_state: DummyState, next_state: DummyState, action: dict[str, str]):
        kore_delta = next_state.kore_me - previous_state.kore_me

        waiting = 'None' in action.values()

        reward = max(kore_delta, 0) + waiting * -100

        return reward
