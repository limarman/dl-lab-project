import re

from kaggle_environments.envs.kore_fleets.helpers import *
import numpy as np
from kaggle_environments.envs.kore_fleets.kore_fleets import get_col_row, get_to_pos
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

    def get_overlapping_kore_map(self) -> ndarray:
        """
        Makes sure that the kore map has the same axis layout as the other maps generated
        :return: kore map 21x21 numpy array
        """
        kore_map = np.zeros((21,21))
        for pos, cell in self.board.cells.items():
            kore_map[pos.x][pos.y] = cell.kore

        return kore_map

    def get_kore_cargo_map(self) -> ndarray:
        """
        Returns map with kore cargo per fleet. Thereby every shipyard has player.kore as value.
        Allied forces get positive sign, enemy forces negative sign
        :return: kore cargo map 21x21
        """
        kore_cargo_map = np.zeros((21,21))
        for shipyard in self.board.players[0].shipyards:
            shipyard_pos = shipyard.position
            kore_cargo_map[shipyard_pos.x, shipyard_pos.y] = self.get_kore_me()

        for shipyard in self.board.players[1].shipyards:
            shipyard_pos = shipyard.position
            kore_cargo_map[shipyard_pos.x, shipyard_pos.y] = -self.get_kore_opponent()

        for fleet in self.board.players[0].fleets:
            fleet_pos = fleet.position
            fleet_cargo = fleet.kore
            kore_cargo_map[fleet_pos.x, fleet_pos.y] = fleet_cargo

        for fleet in self.board.players[1].fleets:
            fleet_pos = fleet.position
            fleet_cargo = fleet.kore
            kore_cargo_map[fleet_pos.x, fleet_pos.y] = -fleet_cargo

        return kore_cargo_map

    def get_ship_map(self) -> ndarray:
        """
        Returns a map with the number of ships (shipyard or fleet) per square
        Allied forces have positive sign, opponent(s) have negative sign
        :return: ship map 21x21 numpy array
        """
        ship_map = np.zeros((21,21))
        for _, shipyard in enumerate(self.board.players[0].shipyards):
            shipyard_pos = shipyard.position
            ship_count = shipyard.ship_count
            ship_map[shipyard_pos.x, shipyard_pos.y] = ship_count

        for _, shipyard in enumerate(self.board.players[1].shipyards):
            shipyard_pos = shipyard.position
            ship_count = shipyard.ship_count
            ship_map[shipyard_pos.x, shipyard_pos.y] = -ship_count

        for _, fleet in enumerate(self.board.players[0].fleets):
            fleet_pos = fleet.position
            ship_count = fleet.ship_count
            ship_map[fleet_pos.x, fleet_pos.y] = ship_count

        for _, fleet in enumerate(self.board.players[1].fleets):
            fleet_pos = fleet.position
            ship_count = fleet.ship_count
            ship_map[fleet_pos.x, fleet_pos.y] = -ship_count

        return ship_map


    def get_shipyard_fleets_map(self) -> ndarray:
        """
        Returns list of length 21*21
        """
        shipyard_fleets_map = np.zeros((21, 21))
        for shipyard_idx, shipyard in enumerate(self.board.players[0].shipyards):
            shipyard_pos = shipyard.position
            ship_count_in_decimals = float(str(f".{shipyard.ship_count}".zfill(3)))
            shipyard_fleets_map[shipyard_pos.x, shipyard_pos.y] = shipyard_idx + 1 + ship_count_in_decimals

        for shipyard_idx, shipyard in enumerate(self.board.players[1].shipyards):
            shipyard_pos = shipyard.position
            ship_count_in_decimals = float(str(f".{shipyard.ship_count}".zfill(3)))
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
        return max([shipyard.max_spawn for shipyard in self.player_me.shipyards])

    def get_max_spawn_map(self) -> ndarray:
        """
        Returns a map of maximum spawns possible per square (per shipyard)
        Thereby a positive sign indicates allied shipyards and a negative sign represents enemy shipyards
        This map can also act as an shipyard indicator (non zero when shipyard)
        :return: max spawn map 21x21 numpy array
        """
        max_spawn_map = np.zeros((21, 21))
        for _, shipyard in enumerate(self.board.players[0].shipyards):
            shipyard_pos = shipyard.position
            max_spawn = shipyard.max_spawn
            max_spawn_map[shipyard_pos.x, shipyard_pos.y] = max_spawn

        for _, shipyard in enumerate(self.board.players[1].shipyards):
            shipyard_pos = shipyard.position
            max_spawn = shipyard.max_spawn
            max_spawn_map[shipyard_pos.x, shipyard_pos.y] = -max_spawn

        return max_spawn_map

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

    def get_shipyards_me_map(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for shipyard in self.player_me.shipyards:
            x_pos, y_pos = shipyard.position
            pos_map[x_pos][y_pos] = shipyard.ship_count
        return pos_map

    def get_shipyards_opponent_map(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for shipyard in self.player_opponent.shipyards:
            x_pos, y_pos = shipyard.position
            pos_map[x_pos][y_pos] = shipyard.ship_count
        return pos_map

    def get_ships_me_map(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for fleet in self.player_me.fleets:
            x_pos, y_pos = fleet.position
            pos_map[x_pos][y_pos] = fleet.ship_count
        return pos_map

    def get_ships_opponent_map(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for fleet in self.player_opponent.fleets:
            x_pos, y_pos = fleet.position
            pos_map[x_pos][y_pos] = fleet.ship_count
        return pos_map

    def get_cargo_me_map(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for fleet in self.player_me.fleets:
            x_pos, y_pos = fleet.position
            pos_map[x_pos][y_pos] = fleet.kore
        return pos_map

    def get_cargo_opponent_map(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for fleet in self.player_opponent.fleets:
            x_pos, y_pos = fleet.position
            pos_map[x_pos][y_pos] = fleet.kore
        return pos_map

    def get_shipyards_id_opponent(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for shipyard in self.player_opponent.shipyards:
            x_pos, y_pos = shipyard.position
            pos_map[x_pos][y_pos] = 1
        return pos_map

    def get_shipyards_id_me(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for shipyard in self.player_me.shipyards:
            x_pos, y_pos = shipyard.position
            pos_map[x_pos][y_pos] = 1
        return pos_map

    def get_max_spawn_me_map(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for shipyard in self.player_me.shipyards:
            x_pos, y_pos = shipyard.position
            pos_map[x_pos][y_pos] = shipyard.max_spawn
        return pos_map

    def get_max_spawn_opponent_map(self) -> ndarray:
        pos_map = np.zeros((21, 21))
        for shipyard in self.player_opponent.shipyards:
            x_pos, y_pos = shipyard.position
            pos_map[x_pos][y_pos] = shipyard.max_spawn
        return pos_map

    def get_step(self, normalized=True) -> float:
        """ Returns the normalized current step """
        step = self.board.observation['step'] + 1
        if normalized:
            step = step / self.board.configuration.episode_steps
        return step

    def get_cargo(self) -> float:
        """
        Returns the number of cargo that is on the fleets
        """
        return sum([fleet.kore for fleet in self.player_me.fleets])

    def get_fleet_count_me(self) -> int:
        return len(self.player_me.fleets)

    def get_fleet_count_opponent(self) -> int:
        return len(self.player_opponent.fleets)

    def get_feature_map_flight_plan_me(self):
        return self._get_feature_map_flight_plan_for_player(self.player_me)

    def get_feature_map_flight_plan_opponent(self):
        return self._get_feature_map_flight_plan_for_player(self.player_opponent)

    def _get_feature_map_flight_plan_for_player(self, player: Player) -> ndarray:
        """
        Approximates the flightplan on a map by indicating the timesteps until a fleet will be there,
        e.g. consider a box-flightplan:
        000000
        018700
        020600
        034500

        The actual values are (50-timesteps)/50 such that a number close to one indicates that a fleet will be
        there soon and a number close to zero symbolizes that in the far future there will be a fleet.

        If multiple fleets will travel over a cell, the minimum number of timesteps is indicated.
        Flightplans end correctly at shipyards, but collisions are not taken into account here.
        Flightplans are approximated for at most 50 steps
        """
        # need to know shipyard positions since flightplans end there
        shipyards = self.board.shipyards.values()
        shipyard_pos = [s.position.x + s.position.y * 21 for s in shipyards]

        feature_map = np.zeros((21,21))

        for fleet in player.fleets:
            directions_numbers_list = self._get_directions_numbers_list(fleet)

            if not directions_numbers_list:
                break

            current_pos = fleet.position.x + fleet.position.y * 21
            pos_list = []

            for _ in range(50):
                if directions_numbers_list:
                    current_elem = directions_numbers_list.pop(0)
                if current_elem == 'C':
                    # new shipyard is created
                    break

                if directions_numbers_list and directions_numbers_list[0].isnumeric():
                    step_length = int(directions_numbers_list.pop(0)) + 1
                    for _ in range(step_length):
                        current_pos = self._get_to_pos_char(current_pos, current_elem)
                        if not current_pos in pos_list:
                            pos_list.append(current_pos)
                else:
                    current_pos = self._get_to_pos_char(current_pos, current_elem)
                    if not current_pos in pos_list:
                        pos_list.append(current_pos)

                if current_pos in shipyard_pos:
                    # remove shipyard pos
                    pos_list.pop()
                    break

            # transform positions into np array (only 50 step approximation)
            feature_map = self._positions_to_map_enumerating(feature_map, pos_list[:50])

        return feature_map

    def _positions_to_map_enumerating(self, feature_map: ndarray, pos_list: List[int]) -> ndarray:
        """
        Maps positions to a feature map, where the first position is represented by one
        and the last by (1/50). The list ist truncated at 50. Hence the formula is (50-i)/50
        """
        coordinate_list = [get_col_row(21, pos) for pos in pos_list]
        coordinate_list = coordinate_list[:50]

        def get_val(pos: int):
            return (50 - (pos + 1)) / 50

        for idx, (col, row) in enumerate(coordinate_list):
            current_val = feature_map[col][row]
            if current_val == 0:
                next_val = get_val(idx)
            else:
                next_val = max(get_val(idx), current_val)
            feature_map[col][row] = next_val

        return feature_map



    def _get_directions_numbers_list(self, fleet) -> List[str]:
        """
        Get a decomposition of the flightplan by directions and numbers
        E.g.: W3E12S3 -> ['W', '3', 'E', '12', 'S','3']

        Filters 'C' (stands for build new shipyards) if there are too
        few ships for that in the fleet
        """
        flight_plan = fleet.flight_plan

        if not flight_plan:
            return []

        if 'C' in flight_plan and fleet.ship_count < self.board.configuration.convert_cost:
            # remove 'C' = build shipyard since it will be ignored by the environment
            flight_plan = flight_plan.replace('C', '')

        # decompose the flightplan into list of N,S,W,E,C and numbers, e.g. [N,4,W,12,S,22]
        direction_number_list = re.split('(\d+ |[N,S,W,E,C])', flight_plan)
        direction_number_list = list(filter(None, direction_number_list))

        if direction_number_list and direction_number_list[0].isnumeric():
            # flightplan is missing the current direction
            # in case flightplan starts with a number
            # we need to add the direction but decrease the number
            if int(direction_number_list[0]) > 1:
                current_direction = fleet.direction.name[0]
                first_dir_length = str(int(flight_plan[0]) - 1)
                direction_number_list = [current_direction] + [str(first_dir_length)] + direction_number_list[1:]
            else:
                # simply remove first number
                direction_number_list = direction_number_list[1:]

        return direction_number_list


    def _get_to_pos_char(self, current_pos: int, direction: str) -> int:
        """
        Calculate the next number representing the direction change

        Numbers in [0,21*21-1] are representing positions (x,y) in the grid 21 x 21
        by the translation: x = pos % size, y = pos // size

        Adapted from the kaggle environment helper functions
        """
        size = 21
        col, row = get_col_row(21, current_pos)
        if direction == "S":
            return current_pos - size if current_pos >= size else size ** 2 - size + col
        elif direction == "N":
            return col if current_pos + size >= size ** 2 else current_pos + size
        elif direction == "E":
            return current_pos + 1 if col < size - 1 else row * size
        elif direction == "W":
            return current_pos - 1 if col > 0 else (row + 1) * size - 1


    def game_result(self) -> int:
        """
        Derives from the board state an indicator for the winning player:
        1: Player is winning
        0: Opponent wins/Game undecided yet/Draw/
        """
        if self.get_step(normalized=True) == 1:
            if self.get_kore_me() > self.get_kore_opponent():
                return 1
            return 0
        if self.get_shipyard_count_opponent() == 0 and self.get_fleet_count_opponent() == 0:
            return 1

        return 0

