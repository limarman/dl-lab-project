import math
from typing import Dict

from src.Rewards.kore_reward import KoreReward
from src.States.advanced_state import AdvancedState


class AdvancedReward(KoreReward):
    """
    Simple reward implementation for quick prototyping
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """

    @staticmethod
    def get_reward_from_action(current_state: AdvancedState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = current_state.apply_action_to_board(actions)

        return AdvancedReward.get_reward_from_states(current_state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: AdvancedState, next_state: AdvancedState):
        kore_delta = max(0, next_state.kore_me - previous_state.kore_me)
        ship_delta = max(0, next_state.ship_count_me - previous_state.ship_count_me)
        kore_distance = max(0, kore_delta - (next_state.kore_opponent - previous_state.kore_opponent))
        ship_distance = max(0, ship_delta - (next_state.ship_count_opponent - previous_state.ship_count_opponent))
        shipyard_delta = max(0, next_state.shipyard_count_me - previous_state.shipyard_count_me)
        ships_destroyed = max(0, next_state.ship_count_opponent - previous_state.ship_count_opponent)
        cargo_delta = max(0, next_state.cargo - previous_state.cargo)

        # increasing ships is more valuable in the beginning than the end
        # TODO define more advanced reward function
        # weight_decay_ships = math.exp(1 - previous_state.step_normalized) * (1 / math.exp(1))
        weight_decay = 1 - next_state.step_normalized
        #reward = kore_delta + cargo + weight_decay * ship_delta * 10 + 0.2 * kore_distance \
        #         + 10 * ships_destroyed + 20 * shipyard_delta

        reward = kore_delta + cargo_delta + 10 * ships_destroyed + 50 ** next_state.step_normalized \
                 + ship_delta * 10 + ship_distance * 10

        return reward

    @staticmethod
    def get_reward(previous_state: AdvancedState, next_state: AdvancedState, action: Dict[str, str]):
        return AdvancedReward.get_reward_from_states(previous_state, next_state)
