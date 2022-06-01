from kaggle_environments.envs.kore_fleets.helpers import *
import numpy as np
from numpy import ndarray


class BoardWrapper:
    """
    Wrapper for the Board class from the Kaggle environment.
    Provides utilities to easier generate input features.
    """

    def __init__(self, board: Board, player_id=None):
        """
        Initializes the current player and its opponent. Since the
        BoardWrapper may also be used in modelling the opponent,
        a player_id can be specified, otherwise the current player
        will be our player

        :param board: as returned by the Kaggle env
        :param player_id: valid player id of our player
        """
        self.board = board
        if (len(board.opponents)) > 1:
            raise Exception('BoardWrapper only supports 2 Player '
                            + 'game but current game has more than two players')
        if player_id:
            if not board.players[player_id]:
                raise Exception('Invalid Player ID')
            self.player_me = board.players[player_id]
            self.player_opponent = board.players.values().remove(self.player_me)[0]
        else:
            self.player_me = board.current_player
            if board.opponents:
                self.player_opponent = board.opponents.pop()

    def get_kore_map(self) -> [float]:
        """
        Returns list of length 21*21
        """
        return [cell.kore for pos, cell in self.board.cells.items()]

    def get_shipyard_fleets_map(self) -> ndarray:
        """
        Returns list of length 21*21
        """
        shipyard_fleets_map = np.zeros((21, 21))
        for shipyard_idx, shipyard in enumerate(self.board.players[0].shipyards):
            shipyard_pos = shipyard.position
            ship_count_in_decimals = float(str(f".{shipyard.ship_count}"))
            shipyard_fleets_map[shipyard_pos.x, shipyard_pos.y] = shipyard_idx + 1 + ship_count_in_decimals

        for shipyard_idx, shipyard in enumerate(self.board.players[1].shipyards):
            shipyard_pos = shipyard.position
            ship_count_in_decimals = float(str(f".{shipyard.ship_count}"))
            shipyard_fleets_map[shipyard_pos.x, shipyard_pos.y] = -(shipyard_idx + 1 + ship_count_in_decimals)

        for fleet_idx, fleet in enumerate(self.board.players[0].fleets):
            fleet_pos = fleet.position
            ship_count = fleet.ship_count
            shipyard_fleets_map[fleet_pos.x, fleet_pos.y] = ship_count

        for fleet_idx, fleet in enumerate(self.board.players[1].fleets):
            fleet_pos = fleet.position
            ship_count = fleet.ship_count
            shipyard_fleets_map[fleet_pos.x, fleet_pos.y] = -ship_count

        return shipyard_fleets_map

    def get_kore_me(self) -> float:
        return self.player_me.kore

    def get_kore_opponent(self) -> float:
        return self.board.opponents.pop().kore

    def get_max_spawn_me(self) -> int:
        # TODO adjust to multiple shipyards
        if self.player_me.shipyards:
            return self.player_me.shipyards[0].max_spawn
        return 0

    def get_max_spawn_shipyard(self, shipyard_idx: int) -> int:
        n_shipyards = len(self.get_shipyards_of_current_player())
        if n_shipyards == 0:
            return 0
        if shipyard_idx >= n_shipyards:
            shipyard_idx = n_shipyards - 1
        return self.player_me.shipyards[shipyard_idx].max_spawn

    def get_ship_count_shipyard(self, shipyard_idx):
        n_shipyards = len(self.get_shipyards_of_current_player())
        if n_shipyards == 0:
            return 0
        if shipyard_idx >= n_shipyards:
            shipyard_idx = n_shipyards - 1
        return self.player_me.shipyards[shipyard_idx].ship_count

    def get_ship_count_me(self) -> int:
        return self._get_ship_count(self.player_me)

    def get_ship_count_opponent(self) -> int:
        return self._get_ship_count(self.player_opponent)

    def get_shipyard_count_me(self):
        return len(self.player_me.shipyards)

    def get_shipyard_count_opponent(self):
        return len(self.player_opponent.shipyards)

    def _get_ship_count(self, player: Player) -> int:
        return sum([fleet.ship_count for fleet in player.fleets]) \
               + sum(shipyard.ship_count for shipyard in player.shipyards)

    def get_shipyards_of_current_player(self) -> List[Shipyard]:
        """
        Returns a list of shipyards of the player who plays the current turn

        :return list of type shipyard
        """

        if self.board.current_player.shipyards is None:
            return []
        return self.board.current_player.shipyards

    def get_fleet_pos(self) -> np.ndarray:
        """
        Returns a grid, where the fleets are represented by positive number of ships for current player
        and by the negative number of ships for the opponent respectively

        :return np array (board_size, board_size)
        """
        return self._get_pos(self.board.fleets)

    def get_shipyard_pos(self) -> np.ndarray:
        """
        Returns a grid, where the fleets are represented by positive
        number of ships for current player and by the negative number
        of ships for the opponent respectively

        :return np array (board_size, board_size)
        """
        return self._get_pos(self.board.shipyards)

    def _get_pos(self, shipyards_or_fleets) -> np.ndarray:
        """
        Returns a grid, where the fleets/shipyards are represented by positive number of ships for current player
        and by the negative number of ships for the opponent respectively

        :return np array (board_size, board_size)
        """
        pos_map = np.zeros((21, 21))
        for _, entity in shipyards_or_fleets.items():
            x_pos, y_pos = entity.position
            if entity.player == self.board.current_player:
                sign = 1
            else:
                sign = -1

            pos_map[x_pos][y_pos] = sign * (1 + entity.ship_count)

        return pos_map

    def get_step(self, normalized=True) -> float:
        """ Returns the normalized current step """
        step = self.board.observation['step']
        if normalized:
            step = step / self.board.configuration.episode_steps
        return step

    def get_cargo(self) -> float:
        """
        Returns the number of cargo that is on the fleets
        """
        return sum([fleet.kore for fleet in self.player_me.fleets])

