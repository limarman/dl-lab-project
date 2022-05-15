import unittest
import numpy as np
from unittest.mock import MagicMock

from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board

from src.rewards.SimpleReward import SimpleReward
from src.states.BoardWrapper import BoardWrapper
from src.states.SimpleState import SimpleState


class TestSimpleReward(unittest.TestCase):

    def setUp(self):
        self.board = create_board(starting_kore=500)
        self.board_wrapper = BoardWrapper(self.board)
        self.simple_reward = SimpleReward()
        self.simple_state = SimpleState(self.board)
        self.spawn_ship_action = ShipyardAction.spawn_ships(1)

    def test_apply_action_to_board(self):
        next_state = self.simple_reward.apply_action_to_board(self.simple_state, [self.spawn_ship_action])

        # test that there is one ship more
        ship_count_previous = self.simple_state.ship_count_me
        ship_count_current = next_state.ship_count_me
        self.assertEqual(ship_count_current - ship_count_previous, 1)

        # test that the kore value is reduced by 10 (= ship price)
        kore_previous = self.simple_state.kore_me
        kore_current = next_state.kore_me
        self.assertEqual(kore_previous - kore_current, 10)




