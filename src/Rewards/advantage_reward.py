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

    def __init__(self, ship_value: int = 5, shipyard_value: int = 20):
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
        self.__spawn_value = ship_value

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
        return self.get_advantage(next_state), {}

    def get_advantage(self, kore_state: KoreState) -> float:
        kore_advantage = self.__get_kore_advantage(kore_state)
        ship_advantage = self.__get_ship_advantage(kore_state)
        spawn_advantage = self.__get_spawn_advantage(kore_state)
        cargo_advantage = self.__get_cargo_advantage(kore_state)

        mean_advantage = (kore_advantage + ship_advantage + spawn_advantage + cargo_advantage) / 4

        return mean_advantage

    def __get_cargo_advantage(self, kore_state: KoreState) -> float:
        player_cargo = kore_state.board_wrapper.get_cargo_me()
        opponent_cargo = kore_state.board_wrapper.get_cargo_opponent()
        total_cargo = player_cargo + opponent_cargo

        return player_cargo / (total_cargo + 0.0001)

    def __get_kore_advantage(self, kore_state: KoreState) -> float:
        player_kore = kore_state.board_wrapper.get_kore_me()
        opponent_kore = kore_state.board_wrapper.get_kore_opponent()
        total_kore = player_kore + opponent_kore

        return player_kore / (total_kore + 0.0001)

    def __get_ship_advantage(self, kore_state: KoreState) -> float:
        player_ship_count = kore_state.board_wrapper.get_ship_count_me()
        opponent_ship_count = kore_state.board_wrapper.get_ship_count_opponent()
        total_ship_count = player_ship_count + opponent_ship_count

        return player_ship_count / (total_ship_count + 0.0001)

    def __get_spawn_advantage(self, kore_state: KoreState) -> float:
        player_id = kore_state.board_wrapper.board.current_player.id
        player_shipyards = kore_state.board_wrapper.get_shipyards_of_player(player_id)
        player_spawn = 0
        for shipyard in player_shipyards:
            player_spawn += shipyard.max_spawn

        opponent_id = kore_state.board_wrapper.player_opponent.id
        opponent_shipyards = kore_state.board_wrapper.get_shipyards_of_player(opponent_id)
        opponent_spawn = 0
        for shipyard in opponent_shipyards:
            opponent_spawn += shipyard.max_spawn

        total_spawn = player_spawn + opponent_spawn

        return player_spawn / (total_spawn + 0.0001)

    def __get_player_value(self, kore_state: KoreState):
        player_kore_value = kore_state.board_wrapper.get_kore_me()

        player_ship_count = kore_state.board_wrapper.get_ship_count_me()
        player_ship_value = player_ship_count * (self.KORE_PER_SHIP + self.__ship_value)

        kore_state.board_wrapper.get_shipyards_of_current_player()

        player_id = kore_state.board_wrapper.board.current_player.id
        player_shipyards = kore_state.board_wrapper.get_shipyards_of_player(player_id)
        player_shipyards_value = 0
        for shipyard in player_shipyards:
            player_shipyards_value += self.__shipyard_value + shipyard.max_spawn * self.__spawn_value

        return player_kore_value + player_ship_value + player_shipyards_value

    def __get_opponent_value(self, kore_state: KoreState):
        opponent_kore_value = kore_state.board_wrapper.get_kore_opponent()

        opponent_ship_count = kore_state.board_wrapper.get_ship_count_opponent()
        opponent_ship_value = opponent_ship_count * (self.KORE_PER_SHIP + self.__ship_value)

        opponent_id = kore_state.board_wrapper.board.opponents[0].id
        opponent_shipyards = kore_state.board_wrapper.get_shipyards_of_player(opponent_id)
        opponent_shipyards_value = 0
        for shipyard in opponent_shipyards:
            opponent_shipyards_value += self.__shipyard_value + shipyard.max_spawn * self.__spawn_value
        return opponent_kore_value + opponent_ship_value + opponent_shipyards_value

    def reset(self):
        pass
