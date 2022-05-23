from kaggle_environments.envs.kore_fleets.helpers import *
import numpy
import random

'''
Class for rule based actions as farming, attacking, building etc
'''


class RuleBasedActor:

    def __init__(self, board: Board):
        self.board = board

    '''
    Chooses a random point inside of given ring to expand to (inside radius not inside non_expand_margin)
    creates a box path flight plan with expanding
    '''

    def expand_randomly(self, shipyard: Shipyard, radius: int, non_expand_margin: int) -> ShipyardAction:

        assert radius > non_expand_margin, 'For expanding it has to be radius>non_expand_margin'

        # sampling at most 1000 times a random position with given radius around shipyard
        for _ in range(0, 1000):

            off_x = random.randint(0, 2* radius) - radius
            off_y = random.randint(0, 2* radius) - radius
            off = Point(off_x, off_y)

            if abs(off_x) <= non_expand_margin or abs(off_y) <= non_expand_margin:
                continue

            print(f"Expand coordinates: {off_x}, {off_y}")
            # check whether at that position there is already some shipyard
            if not self.board.get_shipyard_at_point(shipyard.position.translate(off, self.board.configuration.size)):
                fleet_size = min(100, shipyard.ship_count)
                return ShipyardAction.launch_fleet_with_flight_plan(fleet_size,
                                                                    self._get_expand_box_flight_plan(off_x, off_y))

        return None

    '''
    Searches for the best box farmer (most kore on the path per time step, ignoring the regeneration)
    Starting the farm flight plan from the given shipyard
    Searching in a box with size 2 * radius around the shipyard, whereas the radius should not exceed 9
    '''

    def start_optimal_box_farmer(self, shipyard: Shipyard, radius: int) -> ShipyardAction:

        kore_map = self._kore_on_paths_map(shipyard, radius)
        self._normalize_by_step_count(kore_map, radius)
        max_x, max_y = self._argmax_of_2dim_square(kore_map, 2 * radius + 1)
        flight_plan = self._get_box_farmer_flight_plan(max_x - radius, radius - max_y)
        return ShipyardAction.launch_fleet_with_flight_plan(21, flight_plan)

    '''
    Creating the action to build as many ships as possible at that current time
    '''

    def build_max(self, shipyard: Shipyard, config: Configuration) -> ShipyardAction:
        max_spawn = shipyard.max_spawn
        kore = shipyard.player.kore
        ship_cost = config.spawn_cost

        # create action
        number_of_ships_to_create = min(max_spawn, kore / ship_cost)

        return ShipyardAction.spawn_ships(number_of_ships_to_create)

    '''
    Launches all ships currently in the shipyard to attack the closest enemy shipyard
    '''

    def attack_closest(self, shipyard: Shipyard) -> ShipyardAction:

        no_ships = shipyard.ship_count
        enemy_shipyard = self._get_closest_enemy_shipyard(self.board, shipyard.position, shipyard.player)
        flight_plan = self._get_shortest_flight_path_between(shipyard.position, enemy_shipyard.position,
                                                             self.board.configuration.size)

        return ShipyardAction.launch_fleet_with_flight_plan(no_ships, flight_plan)

    '''
    Creates a "kore paths map" with given radius and shipyard as the center
    Thereby one entry of the map stands for the number of kore on the path including this point as the furthest point 
    from the shipyard.
    '''

    def _kore_on_paths_map(self, shipyard: Shipyard, radius: int) -> numpy.ndarray:

        map = numpy.zeros(shape=(2 * radius + 1, 2 * radius + 1))

        shipyard_map_pos_x = radius
        shipyard_map_pos_y = radius

        shipyard_origin = Point(shipyard_map_pos_x, shipyard_map_pos_y)

        axes = [Point(0, -1),
                Point(1, 0),
                Point(0, 1),
                Point(-1, 0)]

        # filling the axes
        for p in axes:
            for i in range(1, radius + 1):
                map[self._flip(shipyard_origin + p * i)] = \
                    self.board.get_cell_at_point(
                        shipyard.position.translate(self._toggle_translation_space(p * i),
                                                    self.board.configuration.size)).kore + \
                    self.board.get_cell_at_point(
                        shipyard.position.translate(self._toggle_translation_space(p * (i - 1)),
                                                    self.board.configuration.size)).kore + \
                    map[self._flip(shipyard_origin + p * (i - 1))]

        # fill the rest of the map
        # axis vectors for every quadrant
        offsets = [(Point(0, -1), Point(1, 0)),
                   (Point(1, 0), Point(0, 1)),
                   (Point(0, 1), Point(-1, 0)),
                   (Point(-1, 0), Point(0, -1))]

        for (off_y, off_x) in offsets:
            # bringing the pointer in correct start position
            start_pos = Point(shipyard_map_pos_x, shipyard_map_pos_y)

            for x in range(1, radius + 1):
                for y in range(1, radius + 1):
                    translation = (off_x * x) + (off_y * y)
                    next_pos = start_pos + translation
                    cellpos1 = shipyard.position.translate(
                        self._toggle_translation_space(translation), self.board.configuration.size)
                    cellpos2 = shipyard.position.translate(
                        self._toggle_translation_space(translation - off_y - off_x), self.board.configuration.size)
                    map[self._flip(next_pos)] = \
                        map[self._flip(next_pos - off_y)] + map[self._flip(next_pos - off_x)] - map[
                            self._flip(next_pos - off_x - off_y)] + \
                        self.board.get_cell_at_point(
                            cellpos1).kore - \
                        self.board.get_cell_at_point(
                            cellpos2).kore

        return map

    '''
    Takes the kore path map with the shipyard being the center and normalizes by the number of time steps it takes
    until the fleet returns home with loot (divide by number of steps + 1 (construction cost))
    '''

    def _normalize_by_step_count(self, kore_map: numpy.ndarray, radius: int):
        for x in range(0, 2 * radius + 1):
            for y in range(0, 2 * radius + 1):
                dist_x = abs(x - radius)
                dist_y = abs(y - radius)

                kore_map[y, x] = kore_map[y, x] / (2 * dist_x + 2 * dist_y + 1)

    '''
    Toggles a translation between board coordinate system (north is decrementing y) and kore path map coordinate system
    (north is incrementing y)
    '''

    def _toggle_translation_space(self, translation: Point) -> Point:
        return Point(translation.x, -translation.y)

    '''
    Returns a point which is flipped in x and y
    '''

    def _flip(self, point: Point) -> Point:
        return Point(point.y, point.x)

    '''
    Returns a tuple pointing to a position in the 2 dim square matrix which has a highest entry
    note that the output is to be interpreted as coordinates, thus flipping the dimensions from usual matrix notation
    '''

    def _argmax_of_2dim_square(self, array: numpy.ndarray, size: int) -> Tuple[int, int]:
        max_indices_1d = numpy.argmax(array.flatten())
        res = (max_indices_1d % size, int(max_indices_1d / size))
        return res

    def _get_expand_box_flight_plan(self, off_x_furthest_point: int, off_y_furthest_point: int) -> str:
        flight_plan = ""

        if off_x_furthest_point > 0:
            flight_plan += "E"
            flight_plan += str(abs(off_x_furthest_point) - 1)
        elif off_x_furthest_point < 0:
            flight_plan += "W"
            flight_plan += str(abs(off_x_furthest_point) - 1)

        if off_y_furthest_point > 0:
            flight_plan += "N"
            flight_plan += str(abs(off_y_furthest_point) - 1)
        elif off_y_furthest_point < 0:
            flight_plan += "S"
            flight_plan += str(abs(off_y_furthest_point) - 1)

        flight_plan += "C"

        if off_x_furthest_point > 0:
            flight_plan += "W"
            flight_plan += str(abs(off_x_furthest_point) - 1)
        elif off_x_furthest_point < 0:
            flight_plan += "E"
            flight_plan += str(abs(off_x_furthest_point) - 1)

        if off_y_furthest_point > 0:
            flight_plan += "S"
        elif off_y_furthest_point < 0:
            flight_plan += "N"

        return flight_plan

    '''
    Returns a flight plan for a certain translation on the kore paths map. Thereby the following scheme is chosen:
    - First decide whether North or South
    - After that decide whether West or East
    - conclude the box
    '''

    def _get_box_farmer_flight_plan(self, off_x_furthest_point: int, off_y_furthest_point: int) -> str:
        flight_plan = ""

        if off_x_furthest_point > 0:
            flight_plan += "E"
            flight_plan += str(abs(off_x_furthest_point) - 1)
        elif off_x_furthest_point < 0:
            flight_plan += "W"
            flight_plan += str(abs(off_x_furthest_point) - 1)

        if off_y_furthest_point > 0:
            flight_plan += "N"
            flight_plan += str(abs(off_y_furthest_point) - 1)
        elif off_y_furthest_point < 0:
            flight_plan += "S"
            flight_plan += str(abs(off_y_furthest_point) - 1)

        if off_x_furthest_point > 0:
            flight_plan += "W"
            flight_plan += str(abs(off_x_furthest_point) - 1)
        elif off_x_furthest_point < 0:
            flight_plan += "E"
            flight_plan += str(abs(off_x_furthest_point) - 1)

        if off_y_furthest_point > 0:
            flight_plan += "S"
        elif off_y_furthest_point < 0:
            flight_plan += "N"

        return flight_plan

    '''
    Returns the closest enemy shipyard from given position
    '''

    def _get_closest_enemy_shipyard(self, position: Point, player: Player) -> Shipyard:
        min_dist = 1000000
        enemy_shipyard = None
        for shipyard in self.board.shipyards.values():
            if shipyard.player_id == player.id:
                continue
            dist = position.distance_to(shipyard.position, self.board.configuration.size)
            if dist < min_dist:
                min_dist = dist
                enemy_shipyard = shipyard
        return enemy_shipyard

    '''
    Returns the shortest flight plan possible for the shortest path from position_a to position_b
    '''

    def _get_shortest_flight_path_between(self, position_a, position_b, size, trailing_digits=False) -> str:
        mag_x = 1 if position_b.x > position_a.x else -1
        abs_x = abs(position_b.x - position_a.x)
        dir_x = mag_x if abs_x < size / 2 else -mag_x
        mag_y = 1 if position_b.y > position_a.y else -1
        abs_y = abs(position_b.y - position_a.y)
        dir_y = mag_y if abs_y < size / 2 else -mag_y
        flight_path_x = ""
        if abs_x > 0:
            flight_path_x += "E" if dir_x == 1 else "W"
            flight_path_x += str(abs_x - 1) if (abs_x - 1) > 0 else ""
        flight_path_y = ""
        if abs_y > 0:
            flight_path_y += "N" if dir_y == 1 else "S"
            flight_path_y += str(abs_y - 1) if (abs_y - 1) > 0 else ""
        if not len(flight_path_x) == len(flight_path_y):
            if len(flight_path_x) < len(flight_path_y):
                return flight_path_x + (flight_path_y if trailing_digits else flight_path_y[0])
            else:
                return flight_path_y + (flight_path_x if trailing_digits else flight_path_x[0])
        return flight_path_y + (flight_path_x if trailing_digits or not flight_path_x else flight_path_x[
            0]) if random() < .5 else flight_path_x + (
            flight_path_y if trailing_digits or not flight_path_y else flight_path_y[0])
