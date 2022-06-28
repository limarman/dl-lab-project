from typing import Dict

from src.Rewards.kore_reward import KoreReward
from src.States.kore_state import KoreState


class GameLengthProportionalWinReward(KoreReward):
    """
    Reward which rewards fast wins or in the case of a lose, a long defense
    to encourage the agent to improve even if it is losing a lot
    Making the assumption that long games correspond to a good play
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """

    @staticmethod
    def get_reward_from_action(current_state: KoreState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes
        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = current_state.apply_action_to_board(actions)

        return GameLengthProportionalWinReward.get_reward_from_states(current_state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: KoreState, next_state: KoreState):

        return GameLengthProportionalWinReward.get_reward(previous_state, next_state)

    @staticmethod
    def get_reward(previous_state: KoreState, next_state: KoreState, action: Dict[str, str]):
        """
        In the case of the win the reward is 1000 - number_of_steps
        In the case of a lose the reward is number_of_steps
        :param previous_state:
        :param next_state:
        :param action:
        :return: a reward in the range [0,400] for a lose and [600, 1000] for a win, or 0 if the game is not finished yet
        """

        me_is_busted = True if next_state.board_wrapper.get_shipyard_count_me() == 0 and next_state.board_wrapper.get_ship_count_me() == 0 else False
        opp_is_busted = True if next_state.board_wrapper.get_shipyard_count_opponent() == 0 and next_state.board_wrapper.get_ship_count_opponent() == 0 else False
        step = next_state.board_wrapper.get_step(normalized=False)

        if me_is_busted:
            return step + 1
        elif opp_is_busted:
            return 1000 - (step + 1)
        elif step == 399:
            if next_state.board_wrapper.get_kore_me() > next_state.board_wrapper.get_kore_opponent():
                return 1000 - (step + 1)
            elif next_state.board_wrapper.get_kore_me() < next_state.board_wrapper.get_kore_opponent():
                return step + 1

        #if not end of the game
        return 0
