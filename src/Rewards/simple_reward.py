import math

from src.Rewards.kore_reward import KoreReward
from src.States.simple_state import SimpleState


class SimpleReward(KoreReward):
    """
    Simple reward implementation for quick prototyping
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """

    @staticmethod
    def get_reward_from_action(state: SimpleState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = state.apply_action_to_board(actions)

        return SimpleReward.get_reward_from_states(state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: SimpleState, next_state: SimpleState):
        kore_delta = next_state.kore_me - previous_state.kore_me
        ship_delta = next_state.ship_count_me - previous_state.ship_count_me
        kore_distance = max(0, kore_delta - (next_state.kore_opponent - previous_state.kore_opponent))
        shipyard_delta = max(0,next_state.shipycard_count_me - previous_state.shipycard_count_me)
        shipyards_destroyed = max(0, next_state.ship_count_opponent - previous_state.ship_count_opponent)

        # increasing ships is more valuable in the beginning than the end
        # TODO define more advanced reward function
        weight_decay_ships = math.exp(1 - previous_state.step_normalized) * (1 / math.exp(1))
        reward = kore_delta + weight_decay_ships * ship_delta * 10 + 0.2 * kore_distance + 20 * (shipyards_destroyed + shipyard_delta)

        return reward
