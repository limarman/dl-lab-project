import unittest

import numpy as np
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board

from src.Rewards.SimpleReward import SimpleReward
from src.States.BoardWrapper import BoardWrapper
from src.States.SimpleState import SimpleState


class TestSimpleReward(unittest.TestCase):

    def setUp(self):
        self.board = create_board(starting_kore=500)
        self.board_wrapper = BoardWrapper(self.board)
        self.simple_reward = SimpleReward()
        self.simple_state = SimpleState(self.board)
        self.spawn_ship_action = ShipyardAction.spawn_ships(1)

    def test_reward_from_action(self):
        reward = self.simple_reward.get_reward_from_action(self.simple_state, [self.spawn_ship_action])
        self.assertEqual(reward, 0)


