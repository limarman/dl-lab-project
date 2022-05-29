import unittest

from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board, ShipyardAction

from src.States.advanced_state import AdvancedState


class TestSimpleState(unittest.TestCase):

    def setUp(self):
        self.board = create_board(starting_kore=500)
        self.simple_state = AdvancedState(self.board)
        self.spawn_ship_action = ShipyardAction.spawn_ships(1)

    def test_get_tensor(self):
        tensor_shape = self.simple_state.tensor.shape
        self.assertEqual(list(tensor_shape), [3*21*21 + 8*1])

    def test_apply_action_to_board(self):
        next_state = self.simple_state.apply_action_to_board([self.spawn_ship_action])

        # test that there is one ship more
        ship_count_previous = self.simple_state.ship_count_me
        ship_count_current = next_state.ship_count_me
        self.assertEqual(ship_count_current - ship_count_previous, 1)

        # test that the kore value is reduced by 10 (= ship price)
        kore_previous = self.simple_state.kore_me
        kore_current = next_state.kore_me
        self.assertEqual(kore_previous - kore_current, 10)
