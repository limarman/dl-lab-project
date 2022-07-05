from typing import Dict

from src.Rewards.kore_reward import KoreReward
from src.States.kore_state import KoreState


class AdvantageReward(KoreReward):
    """
    Dummy reward implementation for quick prototyping
    """

    KORE_PER_SHIP: int = 10

    SHIPS_PER_SHIPYARD = 50
    KORE_PER_SHIPYARD: int = (KORE_PER_SHIP * SHIPS_PER_SHIPYARD)

    def __init__(self, ship_value: int = 5, shipyard_value: int = 250):
        """
        This reward will measure the advantage we have over our opponent in terms of "resources".
        Resources in this game include kore/ships/shipyards. The reward will convert the ship count and shipyard count
        into kore values. Additionally, it will factor in additional value (in terms of kore/unit)
        for having ships/shipyards because these need time steps to create.

        :param ship_value: additional value per ship
        :param shipyard_value: additional value per shipyard
        """
        super().__init__()
        self.__ship_value = ship_value
        self.__shipyard_value = (self.SHIPS_PER_SHIPYARD * self.__ship_value) + shipyard_value

    @staticmethod
    def get_reward_from_action(current_state: KoreState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = current_state.apply_action_to_board(actions)
        return AdvantageReward.get_reward_from_states(current_state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: KoreState, next_state: KoreState):
        kore_delta = next_state.kore_me - previous_state.kore_me

        return max(kore_delta, 0)

    def get_reward(self, previous_state: KoreState, next_state: KoreState, action: Dict[str, str]):
        my_value = self.__get_player_value(next_state)
        opponent_value = self.__get_opponent_value(next_state)
        total_value = my_value + opponent_value
        advantage = my_value / (total_value + 0.0001)
        return advantage

    def __get_player_value(self, next_state: KoreState):
        player_kore_value = next_state.board_wrapper.get_kore_me()

        player_ship_count = next_state.board_wrapper.get_ship_count_me()
        player_ship_value = player_ship_count * (self.KORE_PER_SHIP + self.__ship_value)

        player_shipyard_count = next_state.board_wrapper.get_shipyard_count_me()
        player_shipyard_value = player_shipyard_count * (self.KORE_PER_SHIPYARD + self.__shipyard_value)
        return player_kore_value + player_ship_value + player_shipyard_value

    def __get_opponent_value(self, next_state: KoreState):
        opponent_kore_value = next_state.board_wrapper.get_kore_opponent()

        opponent_ship_count = next_state.board_wrapper.get_ship_count_opponent()
        opponent_ship_value = opponent_ship_count * (self.KORE_PER_SHIP + self.__ship_value)

        opponent_shipyard_count = next_state.board_wrapper.get_shipyard_count_opponent()
        opponent_shipyard_value = opponent_shipyard_count * (self.KORE_PER_SHIPYARD + self.__shipyard_value)
        return opponent_kore_value + opponent_ship_value + opponent_shipyard_value
