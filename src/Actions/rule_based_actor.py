from kaggle_environments.envs.kore_fleets.helpers import *
import numpy
import random

from kaggle_environments.envs.kore_fleets.kore_fleets import check_location

'''
Class for rule based actions as farming, attacking, building etc
'''


class RuleBasedActor:

    def __init__(self, board: Board):
        self.board = board

    def expand_optimal(self, shipyard: Shipyard, validity_check=False) -> ShipyardAction:
        """
        Applies the expansion strategy from balanced bot
        :param shipyard: shipyard to expand from
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :return: expanding action as ShipyardAction
        """

        if shipyard is None or shipyard.ship_count < 57:
            return None

        elif validity_check:
            return True

        # copied from balanced agent
        start_dir = random.randint(0, 3)
        next_dir = (start_dir + 1) % 4
        best_kore = 0
        best_gap1 = 0
        best_gap2 = 0
        for gap1 in range(5, 15, 3):
            for gap2 in range(5, 15, 3):
                gap2 = random.randint(3, 9)
                diff1 = Direction.from_index(start_dir).to_point() * gap1
                diff2 = Direction.from_index(next_dir).to_point() * gap2
                diff = diff1 + diff2
                pos = shipyard.position.translate(diff, self.board.configuration.size)
                h = check_location(self.board, pos, self.board.current_player)
                if h > best_kore:
                    best_kore = h
                    best_gap1 = gap1
                    best_gap2 = gap2
        gap1 = str(best_gap1)
        gap2 = str(best_gap2)
        flight_plan = Direction.list_directions()[start_dir].to_char() + gap1
        flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
        flight_plan += "C"

        return ShipyardAction.launch_fleet_with_flight_plan(max(50 + 7, int(shipyard.ship_count / 2)), flight_plan)

    def expand_randomly(self, shipyard: Shipyard, radius: int, non_expand_margin: int, validity_check=False) -> ShipyardAction:
        """
        Chooses a random point inside of given ring to expand to (inside radius not inside non_expand_margin)
        creates a box path flight plan with expanding
        :param shipyard: shipyard to expand from
        :param radius: radius around the shipyard to expand to
        :param non_expand_margin: radius around the shipyard not to expand to
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :return: expanding action as ShipyardAction
        """

        assert radius > non_expand_margin, 'For expanding it has to be radius>non_expand_margin'

        if shipyard is None or shipyard.ship_count < 50:
            return None
        elif validity_check:
            return True

        # sampling at most 1000 times a random position with given radius around shipyard
        for _ in range(0, 1000):

            off_x = random.randint(0, 2 * radius) - radius
            off_y = random.randint(0, 2 * radius) - radius
            off = Point(off_x, off_y)

            if abs(off_x) <= non_expand_margin or abs(off_y) <= non_expand_margin:
                continue

            # print(f"Expand coordinates: {off_x}, {off_y}")
            # check whether at that position there is already some shipyard
            if not self.board.get_shipyard_at_point(shipyard.position.translate(off, self.board.configuration.size)):
                fleet_size = min(100, shipyard.ship_count)
                return ShipyardAction.launch_fleet_with_flight_plan(fleet_size,
                                                                    self._get_expand_box_flight_plan(off_x, off_y))

        return None

    def defend_closest(self, shipyard: Shipyard, validity_check = False):
        """
        If there are more than one shipyard: launches half of the ships currently in the shipyard to defend the closest
        own shipyard, else do not do anything
        :param shipyard: shipyard to start defending act from
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :return: defending action to the closest own shipyard as ShipyardAction
        """
        if shipyard is None:
            return None
        no_ships = shipyard.ship_count

        own_shipyard = self._get_closest_shipyard(shipyard.position, shipyard.player)

        if own_shipyard is None:
            return None

        if validity_check:
            return True

        flight_plan = self._get_shortest_flight_path_between(shipyard.position, own_shipyard.position,
                                                             self.board.configuration.size)

        if len(flight_plan) == 0:
            return None

        return ShipyardAction.launch_fleet_with_flight_plan(int(no_ships/2), flight_plan)

    def start_farmer(self, shipyard: Shipyard, validity_check=False) -> ShipyardAction:
        """
        Tries to start a box-farmer, if not possible an axis farmer
        :param shipyard: shipyard from where to start the farmer
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :return:
        """
        if shipyard is None:
            return None
        elif validity_check:
            return True

        if shipyard.ship_count >= 21:
            return self.start_optimal_box_farmer(shipyard, 9)
        elif shipyard.ship_count >= 3:
            return self.start_optimal_axis_farmer(shipyard, 5)
        else:
            return None

    def start_optimal_boomerang_farmer(self, shipyard : Shipyard, radius: int, number_of_ships: int = 21,
                                       validity_check = False, probabilistic = False) -> ShipyardAction:
        """
        Searches for the best boomerang farmer (most kore on the path per time step, ignoring the regeneration)
        Starting the farm flight plan from the given shipyard
        Searching in a box with size 2 * radius around the shipyard, whereas the radius should not exceed 9
        :param shipyard: shipyard to start farmer from
        :param radius: radius around the shipyard to farm
        :param number_of_ships: number of ships the box farmer should consist of (>=21)
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :param probabilistic: whether the optimal position is chosen by an argmax or sampled proportional to value size
        :return: farming action as ShipyardAction
        """

        if shipyard is None or shipyard.ship_count < 21:
            return None
        elif validity_check:
            return True

        assert number_of_ships >= 21, "Error, boomerang farmer has to have at least 21 ships"

        kore_maps = self._kore_on_paths_map_boomerang(shipyard, radius)
        self._normalize_by_step_count(kore_maps[0], radius)
        self._normalize_by_step_count(kore_maps[1], radius)
        if not probabilistic:
            slice_no, max_x, max_y = self._argmax_of_3dim_square(kore_maps, 2 * radius + 1)
        else:
            slice_no, max_x, max_y = self._probabilistic_argmax_of_3dim_square(kore_maps, 2 * radius + 1,
                                                                               pre_transform= lambda x : x**6)

        flight_plan = self._get_boomerang_farmer_flight_plan(max_x - radius, radius - max_y, slice_no == 0)

        return ShipyardAction.launch_fleet_with_flight_plan(number_of_ships, flight_plan)

    def start_optimal_box_farmer(self, shipyard: Shipyard, radius: int, number_of_ships=21, validity_check=False) -> ShipyardAction:
        """
        Searches for the best box farmer (most kore on the path per time step, ignoring the regeneration)
        Starting the farm flight plan from the given shipyard
        Searching in a box with size 2 * radius around the shipyard, whereas the radius should not exceed 9
        :param shipyard: shipyard to start farmer from
        :param radius: radius around the shipyard to farm
        :param number_of_ships: number of ships the box farmer should consist of (>=21)
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :return: farming action as ShipyardAction
        """
        if shipyard is None or shipyard.ship_count < 21:
            return None
        elif validity_check:
            return True

        assert number_of_ships >= 21, "Error, box farmer has to have at least 21 ships"

        kore_map = self._kore_on_paths_map(shipyard, radius)
        self._normalize_by_step_count(kore_map, radius)
        max_x, max_y = self._argmax_of_2dim_square(kore_map, 2 * radius + 1)
        flight_plan = self._get_box_farmer_flight_plan(max_x - radius, radius - max_y)

        return ShipyardAction.launch_fleet_with_flight_plan(number_of_ships, flight_plan)

    def start_optimal_axis_farmer(self, shipyard, radius, validity_check=False):
        """

        :param shipyard:
        :param radius:
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :return:
        """
        if shipyard is None or shipyard.ship_count < 3:
            return None

        elif validity_check:
            return True

        kore_map = self._kore_on_axis_map(shipyard, radius)
        max_x, max_y = self._argmax_of_2dim_square(kore_map, 2 * radius + 1)
        flight_plan = self._get_box_farmer_flight_plan(max_x - radius, radius - max_y)

        return ShipyardAction.launch_fleet_with_flight_plan(3, flight_plan)

    def build_max(self, shipyard: Shipyard, validity_check=False) -> ShipyardAction:
        """
        Creating the action to build as many ships as possible at that current time
        :param shipyard: shipyard to build ships
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :return: building action as ShipyardAction
        """
        if shipyard is None:
            return None

        max_spawn = shipyard.max_spawn
        kore = shipyard.player.kore
        ship_cost = self.board.configuration.spawn_cost

        # create action
        number_of_ships_to_create = min(max_spawn, int(kore / ship_cost))

        if number_of_ships_to_create == 0:
            return None
        elif validity_check:
            return True

        return ShipyardAction.spawn_ships(number_of_ships_to_create)

    def attack_closest(self, shipyard: Shipyard, validity_check=False) -> ShipyardAction:
        """
        Launches all ships currently in the shipyard to attack the closest enemy shipyard
        :param shipyard: shipyard to start attack from
        :param validity_check: only true when called from get_invalid_action_mask; returns True is action is valid
        :return: attacking action to the closest enemy shipyard as ShipyardAction
        """
        if shipyard is None:
            return None
        no_ships = shipyard.ship_count
        if no_ships < 20:
            return None
        enemy_shipyard = self._get_closest_shipyard(shipyard.position, shipyard.player)

        if enemy_shipyard is None:
            return None

        if validity_check:
            return True

        flight_plan = self._get_shortest_flight_path_between(shipyard.position, enemy_shipyard.position,
                                                             self.board.configuration.size)

        if len(flight_plan) == 0:
            return None

        return ShipyardAction.launch_fleet_with_flight_plan(no_ships, flight_plan)

    def wait(self, shipyard: Shipyard, validity_check=False) -> ShipyardAction:
        if validity_check:
            # otherwise the shipyard action evaluates to None (interpreted as False) in the action mask
            return True

        return ShipyardAction.from_str("None")

    def _kore_on_axis_map(self, shipyard: Shipyard, radius: int) -> numpy.ndarray:
        """
        Creates a "kore axis map" with given radius and shipyard as the center
        Thereby one entry of the map stands for the number of kore on the path including this point as the
        furthest point from the shipyard.
        :param shipyard: shipyard as the center of the map
        :param radius: radius around shipyard to investigate
        :return: map with size (2 * radius + 1, 2 * radius + 1)
        """

        kore_map = numpy.zeros(shape=(2 * radius + 1, 2 * radius + 1))

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
                kore_map[self._flip(shipyard_origin + p * i)] = \
                    self.board.get_cell_at_point(
                        shipyard.position.translate(self._toggle_translation_space(p * i),
                                                    self.board.configuration.size)).kore + \
                    self.board.get_cell_at_point(
                        shipyard.position.translate(self._toggle_translation_space(p * (i - 1)),
                                                    self.board.configuration.size)).kore + \
                    kore_map[self._flip(shipyard_origin + p * (i - 1))]

        return kore_map


    def _kore_on_paths_map(self, shipyard: Shipyard, radius: int) -> numpy.ndarray:
        """
        Creates a "kore paths map" with given radius and shipyard as the center
        Thereby one entry of the map stands for the number of kore on the path including this point as the
        furthest point from the shipyard.
        :param shipyard: shipyard as the center of the map
        :param radius: radius around shipyard to investigate
        :return: map with size (2 * radius + 1, 2 * radius + 1)
        """

        kore_map = numpy.zeros(shape=(2 * radius + 1, 2 * radius + 1))

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
                kore_map[self._flip(shipyard_origin + p * i)] = \
                    self.board.get_cell_at_point(
                        shipyard.position.translate(self._toggle_translation_space(p * i),
                                                    self.board.configuration.size)).kore + \
                    self.board.get_cell_at_point(
                        shipyard.position.translate(self._toggle_translation_space(p * (i - 1)),
                                                    self.board.configuration.size)).kore + \
                    kore_map[self._flip(shipyard_origin + p * (i - 1))]

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
                    kore_map[self._flip(next_pos)] = \
                        kore_map[self._flip(next_pos - off_y)] + kore_map[self._flip(next_pos - off_x)] - kore_map[
                            self._flip(next_pos - off_x - off_y)] + \
                        self.board.get_cell_at_point(
                            cellpos1).kore - \
                        self.board.get_cell_at_point(
                            cellpos2).kore

        return kore_map

    def _kore_on_paths_map_boomerang(self, shipyard: Shipyard, radius: int) -> numpy.ndarray:
        """
        Creates a "kore paths map" with given radius and shipyard as the center
        Thereby one entry of the map stands for the number of kore on the path a boomerang farmer would take
        including this point as the furthest point from the shipyard.
        :param shipyard: shipyard as the center of the map
        :param radius: radius around shipyard to investigate
        :return: (3-dim) map with size (2, 2 * radius + 1, 2 * radius + 1)
        """

        kore_map = numpy.zeros(shape=(2 * radius + 1, 2 * radius + 1))

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
                kore_map[self._flip(shipyard_origin + p * i)] = \
                    self.board.get_cell_at_point(
                        shipyard.position.translate(self._toggle_translation_space(p * i),
                                                    self.board.configuration.size)).kore + \
                    self.board.get_cell_at_point(
                        shipyard.position.translate(self._toggle_translation_space(p * (i - 1)),
                                                    self.board.configuration.size)).kore + \
                    kore_map[self._flip(shipyard_origin + p * (i - 1))]


        #For boomerang farmers it matters in which direction the turn is, as this changes the path
        #different than for box-farmers. That is why we check for left turning and right turning boomerangs
        kore_maps = numpy.stack([kore_map, numpy.copy(kore_map)])

        # fill the rest of the map
        # axis vectors for every quadrant
        offsets = [(Point(0, -1), Point(1, 0)),
                   (Point(1, 0), Point(0, 1)),
                   (Point(0, 1), Point(-1, 0)),
                   (Point(-1, 0), Point(0, -1))]

        for i in range(0, 2): # for right turning and left turning boomerang farmers

            #checking right turning boomerangs for i==0 and left turning for i ==1
            boomerang_turn = 1 if i == 0 else -1

            for (off_y, off_x) in offsets:
                # bringing the pointer in correct start position
                start_pos = Point(shipyard_map_pos_x, shipyard_map_pos_y)

                for x in range(1, radius + 1):
                    for y in range(1, radius + 1):
                        translation = (off_x * x) + (off_y * y)
                        next_pos = start_pos + translation
                        if boomerang_turn == 1:
                            cellpos1 = shipyard.position.translate(
                                self._toggle_translation_space(translation), self.board.configuration.size)
                            cellpos2 = shipyard.position.translate(
                                self._toggle_translation_space(translation - off_x), self.board.configuration.size)
                            kore_maps[i, self._flip(next_pos).x, self._flip(next_pos).y] = \
                                kore_maps[i, self._flip(next_pos - off_x).x, self._flip(next_pos - off_x).y] +\
                                self.board.get_cell_at_point(
                                    cellpos1).kore + \
                                self.board.get_cell_at_point(
                                    cellpos2).kore
                        else: #left turning boomerang
                            cellpos1 = shipyard.position.translate(
                                self._toggle_translation_space(translation), self.board.configuration.size)
                            cellpos2 = shipyard.position.translate(
                                self._toggle_translation_space(translation - off_y), self.board.configuration.size)
                            kore_maps[i, self._flip(next_pos).x, self._flip(next_pos).y] = \
                                kore_maps[i, self._flip(next_pos - off_y).x, self._flip(next_pos - off_y).y] + \
                                self.board.get_cell_at_point(
                                    cellpos1).kore + \
                                self.board.get_cell_at_point(
                                    cellpos2).kore

        return kore_maps

    def _normalize_by_step_count(self, kore_map: numpy.ndarray, radius: int):
        """
        Takes the kore path map with the shipyard being the center and normalizes by the number of time steps it takes
        until the fleet returns home with loot (divide by number of steps + 1 (construction cost))
        :param kore_map:
        :param radius:
        :return:
        """
        for x in range(0, 2 * radius + 1):
            for y in range(0, 2 * radius + 1):
                dist_x = abs(x - radius)
                dist_y = abs(y - radius)

                kore_map[y, x] = kore_map[y, x] / (2 * dist_x + 2 * dist_y + 1)

    def _toggle_translation_space(self, translation: Point) -> Point:
        """
        Toggles a translation between board coordinate system (north is decrementing y) and kore path map coordinate system
        (north is incrementing y)
        :param translation:
        :return:
        """
        return Point(translation.x, -translation.y)

    def _flip(self, point: Point) -> Point:
        """
        Returns a point which is flipped in x and y
        :param point:
        :return:
        """
        return Point(point.y, point.x)

    def _argmax_of_2dim_square(self, array: numpy.ndarray, size: int) -> Tuple[int, int]:
        """
        Returns a tuple pointing to a position in the 2 dim square matrix which has a highest entry
        note that the output is to be interpreted as coordinates, thus flipping the dimensions from usual matrix notation
        :param array:
        :param size:
        :return:
        """
        max_indices_1d = numpy.argmax(array.flatten())
        res = (max_indices_1d % size, int(max_indices_1d / size))
        return res

    def _argmax_of_3dim_square(self, array: numpy.ndarray, size: int) -> Tuple[int, int, int]:
        """
            Returns a tuple pointing to a position in the 3 dim square matrix which has a highest entry
            note that the output is to be interpreted as coordinates, thus flipping the dimensions from usual matrix notation
            :param array: 3 dimensional numpy array
            :param size: diameter of one slice of the 3 dim array
            :return: Tuple(slice_no, x, y)
        """
        number_of_slices, _, _ = array.shape
        arr = array.flatten()
        max_indices_1d = numpy.argmax(array.flatten())
        slice_no = int(max_indices_1d / size**2)
        max_indices_1d = max_indices_1d % size**2
        x = max_indices_1d % size
        y = int(max_indices_1d / size)

        return slice_no, x, y

    def _probabilistic_argmax_of_3dim_square(self, array: numpy.ndarray, size: int,
                                             pre_transform = lambda x: x) -> Tuple[int,int,int]:
        """
            Returns a tuple pointing to a position in the 3 dim square matrix which is a sample from the value induced
            probability distribution of the matrix (after the transform)
            note that the output is to be interpreted as coordinates, thus flipping the dimensions from usual matrix notation
            :param array: 3 dimensional numpy array
            :param size: diameter of one slice of the 3 dim array
            :param pre_transform: lambda function which applies to every matrix entry before sampling
            :return: Tuple(slice_no, x, y)        """

        number_of_slices, _, _ = array.shape
        max_indices_1d = pre_transform(array.flatten())
        normalizing_const = numpy.sum(max_indices_1d)
        max_indices_1d = max_indices_1d / normalizing_const
        max_indices_1d = numpy.random.choice(list(range(len(max_indices_1d))), 1, p=max_indices_1d)[0]
        slice_no = int(max_indices_1d / size**2)
        max_indices_1d = max_indices_1d % size**2
        x = max_indices_1d % size
        y = int(max_indices_1d / size)

        return slice_no, x, y


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

    def _get_box_farmer_flight_plan(self, off_x_furthest_point: int, off_y_furthest_point: int) -> str:
        """
        Returns a flight plan for a certain translation on the kore paths map. Thereby the following scheme is chosen:
        - First decide whether North or South
        - After that decide whether West or East
        - conclude the box
        :param off_x_furthest_point:
        :param off_y_furthest_point:
        :return:
        """
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
            if off_y_furthest_point != 0:
                flight_plan += str(abs(off_x_furthest_point) - 1)
        elif off_x_furthest_point < 0:
            flight_plan += "E"
            if off_y_furthest_point != 0:
                flight_plan += str(abs(off_x_furthest_point) - 1)

        if off_y_furthest_point > 0:
            flight_plan += "S"
        elif off_y_furthest_point < 0:
            flight_plan += "N"

        return flight_plan

    def _get_boomerang_farmer_flight_plan(self, off_x_furthest_point: int, off_y_furthest_point: int, turn_right: bool) -> str:
        """
        Returns a flight plan for a boomerang farmer for a certain translation on the kore paths map.
        :param off_x_furthest_point:
        :param off_y_furthest_point:
        :param turn_right: whether we send a right-turning boomerang farmer or a left-turning boomerang farmer
        :return:
        """
        flight_plan = ""

        if off_x_furthest_point == 0 or off_y_furthest_point == 0:
            #degenerated boomerang can be handled with the box farmer
            return self._get_box_farmer_flight_plan(off_x_furthest_point, off_y_furthest_point)

        #possible quadrants (1,1), (1,0), (0,1), (0,0)
        quadrant = (off_x_furthest_point > 0, off_y_furthest_point > 0)

        dist_no_y = abs(off_y_furthest_point) - 1
        dist_no_x = abs(off_x_furthest_point) - 1

        if turn_right:
            if quadrant == (1,1):
                flight_plan += "N"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "E"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "W"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "S"

            elif quadrant == (0,0):
                flight_plan += "S"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "W"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "E"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "N"

            elif quadrant == (1,0):
                flight_plan += "E"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "S"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "N"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "W"

            elif quadrant == (0,1):
                flight_plan += "W"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "N"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "S"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "E"

        else:
            if quadrant == (0,1):
                flight_plan += "N"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "W"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "E"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "S"

            elif quadrant == (1,0):
                flight_plan += "S"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "E"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "W"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "N"

            elif quadrant == (1,1):
                flight_plan += "E"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "N"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "S"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "W"

            elif quadrant == (0,0):
                flight_plan += "W"
                if dist_no_x != 0:
                    flight_plan += str(dist_no_x)
                flight_plan += "S"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "N"
                if dist_no_y != 0:
                    flight_plan += str(dist_no_y)
                flight_plan += "E"

        return flight_plan

    def _get_closest_shipyard(self, position: Point, player: Player, enemy_shipyard = True) -> Shipyard:
        """
        Returns the closest shipyard from given position. In the case of searching for an allied shipyard,
        the shipyard at the given point is ignored
        Implementation vastly taken from the Balanced Bot
        :param position:
        :param player:
        :param enemy_shipyard: whether we are searching for the closest opponent or own shipyard
        :return:
        """
        min_dist = 1000000
        enemy_shipyard = None
        for shipyard in self.board.shipyards.values():
            if enemy_shipyard and shipyard.player_id == player.id:
                continue
            elif not enemy_shipyard and (shipyard.player_id != player.id or shipyard.position == position):
                continue
            dist = position.distance_to(shipyard.position, self.board.configuration.size)
            if dist < min_dist:
                min_dist = dist
                enemy_shipyard = shipyard
        return enemy_shipyard

    def _get_shortest_flight_path_between(self, position_a, position_b, size, trailing_digits=False) -> str:
        """
        Returns the shortest flight plan possible for the shortest path from position_a to position_b
        """
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
            0]) if random.random() < .5 else flight_path_x + (
            flight_path_y if trailing_digits or not flight_path_y else flight_path_y[0])
