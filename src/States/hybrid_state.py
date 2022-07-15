import numpy as np
from kaggle_environments.envs.kore_fleets.helpers import *

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Actions.action_validity_masking import get_action_validity_mask
from src.States.board_wrapper import BoardWrapper
from src.States.kore_state import KoreState


class HybridState(KoreState):
    """
    A state combining map-shaped input and scalar input that can generate input for the HybridNet.
    If recenter is True, the maps are recentered around the current shipyard
    """

    def __init__(self, board: Board, shipyard: Shipyard=None, recenter=True):
        """
        Initializes all relevant state values via the BoardWrapper that may be relevant in calculating the reward

        :param board: board as provided by the Kaggle env
        """
        self.recenter = recenter
        self.board_wrapper = BoardWrapper(board)
        self.kore_map = self.board_wrapper.get_overlapping_kore_map()
        self.kore_me = self.board_wrapper.get_kore_me()
        self.kore_opponent = self.board_wrapper.get_kore_opponent()
        self.ship_count_me = self.board_wrapper.get_ship_count_me()
        self.ship_count_opponent = self.board_wrapper.get_ship_count_opponent()
        self.shipyard_count_me = self.board_wrapper.get_shipyard_count_me()
        self.shipyard_count_opponent = self.board_wrapper.get_shipyard_count_opponent()
        self.step_normalized = self.board_wrapper.get_step()
        self.cargo = self.board_wrapper.get_cargo()
        if shipyard:
            self.shipyard_pos_x = shipyard.position.x
            self.shipyard_pos_y = shipyard.position.y
            self.valid_action_mask = get_action_validity_mask(shipyard, board)
            self.ships_at_shipyard = shipyard.ship_count
            self.max_spawn_at_shipyard = shipyard.max_spawn
        else:
            self.shipyard_pos_x = 5
            self.shipyard_pos_y = 15
            self.valid_action_mask = [0 for _ in range(ActionAdapterRuleBased().N_ACTIONS)]
            self.ships_at_shipyard = 0
            self.max_spawn_at_shipyard = 0

        tensor = self._get_tensor()
        super(HybridState, self).__init__(self.get_input_shape(), tensor, self.board_wrapper)

    def _get_tensor(self) -> Dict[str, np.ndarray]:
        """
        Returns a dict with a map-shaped output for key 'maps': (num_feature_maps, 21, 21)
        and a scalar output for key 'scalars': (num_scalars,)
        """
        state = {}

        maps = self.board_wrapper.get_feature_map_collection()
        if self.recenter:
            maps = self._center_maps_on(maps, self.shipyard_pos_x, self.shipyard_pos_y)
        state['maps'] = maps

        state['scalars'] = np.array([
            self.kore_me,
            self.kore_opponent,
            self.ship_count_me,
            self.ship_count_opponent,
            self.step_normalized,
            self.shipyard_count_me,
            self.shipyard_count_opponent,
            self.shipyard_pos_x,
            self.shipyard_pos_y,
            self.ships_at_shipyard,
            self.max_spawn_at_shipyard
        ] + self.valid_action_mask)

        return state

    @staticmethod
    def get_input_shape() -> Dict[str, Union[Tuple[int, int, int], Tuple[int]]]:
        num_actions = ActionAdapterRuleBased.N_ACTIONS
        shapes = {
            'maps': (15, 21, 21),
            'scalars': (11+num_actions,)
        }

        return shapes

    def _center_maps_on(self, maps: np.ndarray, x: int, y: int):
        """
        Recenters the map around the current shipyard
        """
        map_list = list(maps)  # does this really work?
        recentered_maps_list = []

        for map in map_list:
            recentered_map = np.zeros(map.shape)  # initialize map and start shfting entries
            board_size = self.board_wrapper.board.configuration.size
            map_x = x
            map_y = y
            shift_x = int(board_size / 2) - map_x
            shift_y = int(board_size / 2) - map_y
            for i in range(board_size):
                for j in range(board_size):
                    recentered_map[i][j] = map[(i - shift_x) % board_size][(j - shift_y) % board_size]
            recentered_maps_list.append(recentered_map)

        return np.stack(recentered_maps_list)
