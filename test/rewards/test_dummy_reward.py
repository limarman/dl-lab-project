import unittest
from unittest.mock import Mock

from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board

from src.Rewards.dummy_reward import DummyReward
from src.States.board_wrapper import BoardWrapper
from src.States.dummy_state import DummyState


class TestSimpleReward(unittest.TestCase):

    def setUp(self):
        self.board = create_board(starting_kore=500)
        self.board_wrapper = BoardWrapper(self.board)
        self.dummy_reward = DummyReward()
        self.dummy_state = DummyState(self.board)
        self.dummy_state = DummyState(self.board)
        self.spawn_ship_action = ShipyardAction.spawn_ships(1)

    def test_get_reward_from_states(self):
        # test positive reward
        previous_state = Mock(kore_me=50)
        next_state = Mock(kore_me=80)
        reward = self.dummy_reward.get_reward_from_states(previous_state, next_state)
        self.assertEqual(30, reward)

        # test negative reward
        previous_state = Mock(kore_me=80)
        next_state = Mock(kore_me=50)
        reward = self.dummy_reward.get_reward_from_states(previous_state, next_state)
        self.assertEqual(0, reward)

    def test_reward_from_action(self):
        reward = self.dummy_reward.get_reward_from_action(self.dummy_state, [self.spawn_ship_action])
        self.assertEqual(0, reward)



