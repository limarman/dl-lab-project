import unittest
import numpy as np

from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board

from src.States.BoardWrapper import BoardWrapper


class TestBoardWrapper(unittest.TestCase):

    def setUp(self):
        self.board = create_board(starting_kore=500)
        self.board_wrapper = BoardWrapper(self.board)

    def test_init(self):
        # test invalid player id
        self.assertRaises(Exception, BoardWrapper, self.board, player_id=42)
        # test more than two agents
        self.assertRaises(Exception, BoardWrapper, create_board(agent_count=4))

        # test that the players are correctly set
        self.assertEqual(self.board_wrapper.player_me, self.board.current_player)
        self.assertEqual(self.board_wrapper.player_opponent, self.board.opponents.pop())

    def test_get_kore_map(self):
        kore_map = self.board_wrapper.get_kore_map()
        self.assertEqual(len(kore_map), 21 * 21)

    def test_get_kore_me(self):
        kore_me = self.board_wrapper.get_kore_me()
        self.assertEqual(kore_me, 500)

    def test_get_kore_opponent(self):
        kore_opponent = self.board_wrapper.get_kore_opponent()
        self.assertEqual(kore_opponent, 500)

    def test_get_ship_count(self):
        ship_count = self.board_wrapper.get_ship_count_me()
        self.assertEqual(ship_count, 0)

    def test_get_shipyard_pos(self):
        shipyard_pos = self.board_wrapper.get_shipyard_pos()
        # own shipyard has pos (5,15) and opponent has pos (15,5) at beginning
        expected_pos = np.zeros((21,21))
        expected_pos[5][15] = 1
        expected_pos[15][5] = -1

        self.assertTrue((shipyard_pos == expected_pos).all())

