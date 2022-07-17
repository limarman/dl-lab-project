from typing import Dict

from src.Rewards.kore_reward import KoreReward
from src.States.kore_state import KoreState


class CompetitiveKoreDeltaReward(KoreReward):
    """
    Reward which:
    - Rewards the gain of kore
    - Penalizes adversary gain of kore,
    - Hands out (negative) "trophy" if (losing)winning in less than 400 time steps
    (to compensate for kore not gained, due to early termination)
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.collected_kore = 0
        self.opp_collected_kore = 0

    @staticmethod
    def get_reward_from_action(current_state: KoreState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes
        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = current_state.apply_action_to_board(actions)
        return CompetitiveKoreDeltaReward.get_reward_from_states(current_state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: KoreState, next_state: KoreState):
        kore_delta = next_state.board_wrapper.get_kore_me() - previous_state.board_wrapper.get_kore_me()

        return max(kore_delta, 0)

    def reset(self):
        self.collected_kore = 0
        self.opp_collected_kore = 0

    def get_reward(self, previous_state: KoreState, next_state: KoreState, action: Dict[str, str]):
        me_is_busted = True if next_state.board_wrapper.get_shipyard_count_me() == 0 and \
                               next_state.board_wrapper.get_ship_count_me() == 0 else False
        opp_is_busted = True if next_state.board_wrapper.get_shipyard_count_opponent() == 0 and \
                                next_state.board_wrapper.get_ship_count_opponent() == 0 else False

        rel_step = max(next_state.board_wrapper.get_step(normalized=True), 0.001)

        #Calulating the trophy (proportional to collected kore up to this point and to rest steps of the game)
        if me_is_busted:
            return -(1/rel_step * self.opp_collected_kore - self.opp_collected_kore)
        elif opp_is_busted:
            return 1/rel_step * self.collected_kore - self.collected_kore

        kore_delta = next_state.board_wrapper.get_kore_me() - previous_state.board_wrapper.get_kore_me()
        kore_delta_opp = next_state.board_wrapper.get_kore_opponent() - previous_state.board_wrapper.get_kore_opponent()

        #Parse how many ships have been build
        built_ships = 0

        for action in action.values():
            if action.startswith("SPAWN"):
                spawned_ships = action[6:]
                built_ships += int(spawned_ships)

        ship_build_cost = min(built_ships * 10, int(previous_state.board_wrapper.get_kore_me()))

        self.collected_kore += kore_delta + ship_build_cost
        self.opp_collected_kore += max(kore_delta_opp, 0)

        #reward = max(kore_delta, 0) - max(kore_delta_opp, 0) + waiting * -50
        reward = kore_delta + ship_build_cost - max(kore_delta_opp, 0)

        return reward
