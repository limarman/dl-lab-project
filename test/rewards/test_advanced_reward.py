import unittest

from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board

from src.Rewards.advanced_reward import AdvancedReward
from src.States.board_wrapper import BoardWrapper
from src.States.advanced_state import AdvancedState


class TestAdvancedReward(unittest.TestCase):

    def setUp(self):
        self.board = create_board(starting_kore=500)
        self.board_wrapper = BoardWrapper(self.board)
        self.advanced_reward = AdvancedReward()
        self.advanced_state = AdvancedState(self.board)
        self.spawn_ship_action = ShipyardAction.spawn_ships(1)

