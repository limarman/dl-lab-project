import unittest

from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board

from src.states.SimpleState import SimpleState


class TestSimpleState(unittest.TestCase):

    def setUp(self):
        self.board = create_board(starting_kore=500)
        self.simple_state = SimpleState(self.board)

    def test_get_tensor(self):
        tensor_shape = self.simple_state.tensor.shape
        self.assertEqual(list(tensor_shape), [3*21*21 + 5*1])
