import unittest
import numpy as np
import re

from kaggle_environments.envs.kore_fleets.test_kore_fleets import create_board, ShipyardAction
from kaggle_environments.envs.kore_fleets.kore_fleets import get_to_pos, get_col_row

from src.States.board_wrapper import BoardWrapper


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

    def test_get_feature_maps_flight_plan_box(self):
        self._test_get_feature_maps_flight_plan('W3N3E3S', 50)

    def test_get_feature_maps_flight_plan_new_shipyard(self):
        self._test_get_feature_maps_flight_plan('N3C', 50)

    def test_get_feature_maps_flightplan_too_few_ships(self):
        self._test_get_feature_maps_flight_plan('N3C', 49)

    def _test_get_feature_maps_flight_plan(self, flight_plan, ships):
        """
        Simulate flightplans by the kaggle env and compare it to our calculation
        """
        self._launch_flight_plan(flight_plan, ships)
        board_wrapper = BoardWrapper(self.board)
        res = board_wrapper.get_feature_map_flight_plan_me()
        expected = self._simulate_flight()
        self.assertTrue((res == expected).all())

    def _launch_flight_plan(self, flightplan: str, ships):
        actions = [ShipyardAction.spawn_ships(1)] * ships
        actions.append(ShipyardAction.launch_fleet_with_flight_plan(ships, flightplan))

        for action in actions:
            shipyard = self.board.current_player.shipyards[0]
            shipyard.next_action = action
            self.board = self.board.next()

    def _simulate_flight(self):
        expected = np.zeros((21, 21))

        # simulate board
        for i in range(50):
            self.board = self.board.next()
            for fleet in self.board.current_player.fleets:
                x = fleet.position.x
                y = fleet.position.y

                expected[x][y] = max(expected[x][y], (50-(i+1))/50)
        return expected

    def test_get_to_pos_char(self):
        N = self.board_wrapper._get_to_pos_char(5, 'N')
        W = self.board_wrapper._get_to_pos_char(5, 'W')
        S = self.board_wrapper._get_to_pos_char(5, 'S')
        E = self.board_wrapper._get_to_pos_char(5, 'E')

        self.assertEqual(W, 5-1)
        self.assertEqual(E, 5+1)
        self.assertEqual(N, 5+21)
        # overflow of the board
        self.assertEqual(S, 5+(21*20))








