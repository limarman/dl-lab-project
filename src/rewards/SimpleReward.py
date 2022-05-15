import math
from typing import Dict, List

from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Board

from src.rewards.KoreReward import KoreReward
from src.states.SimpleState import SimpleState


class SimpleReward(KoreReward):
    """
    Simple reward implementation for quick prototyping
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """
        pass

    def get_reward(self, current_state: SimpleState, action) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = self.apply_action_to_board()
        kore_delta = current_state.kore_me - next_state.kore_me
        ship_delta = current_state.ship_count_me - next_state.kore_me

        # increasing ships is more valuable in the beginning than the end
        weight_decay_ships = math.exp(1-current_state.step_normalized) * (1/math.exp(1))
        reward = kore_delta + weight_decay_ships * ship_delta

        # TODO define more advanced reward function

        return reward

    def apply_action_to_board(self, state: SimpleState, actions: List[ShipyardAction]) -> SimpleState:
        """
        Applies the shipyard actions to the current state of the current player

        :param state: current state
        :param actions: actions to be applied to the shipyards
        :return: new state after applying the actions
        """
        for shipyard, action in zip(state.board_wrapper.board.current_player.shipyards, actions):
            shipyard.next_action = action
        next_board = state.board_wrapper.board.next()
        next_state = SimpleState(next_board)

        return next_state

