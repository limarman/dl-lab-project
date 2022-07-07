import numpy as np
from kaggle_environments.envs.kore_fleets.helpers import *

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Actions.action_validity_masking import get_action_validity_mask
from src.States.board_wrapper import BoardWrapper
from src.States.kore_state import KoreState


class HybridState(KoreState):
    """
    A state combining map-shaped input and scalar input that can generate input for the HybridNet
    """

    def __init__(self, board: Board, shipyard: Shipyard=None):
        """
        Initializes all relevant state values via the BoardWrapper that may be relevant in calculating the reward

        :param board: board as provided by the Kaggle env
        """
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
        else:
            self.shipyard_pos_x = 5
            self.shipyard_pos_y = 5
            self.valid_action_mask = [0 for _ in range(ActionAdapterRuleBased().N_ACTIONS)]

        tensor = self._get_tensor()
        super(HybridState, self).__init__(self.get_input_shape(), tensor, self.board_wrapper)

    def _get_tensor(self) -> Dict[str, np.ndarray]:
        """
        Returns a dict with a map-shaped output for key 'maps': (num_feature_maps, 21, 21)
        and a scalar output for key 'scalars': (num_scalars,)
        """
        state = {}

        state['maps'] = np.stack([
            self.board_wrapper.get_ships_me_map(),
            self.board_wrapper.get_ships_opponent_map(),
            self.board_wrapper.get_shipyards_id_me(),
            self.board_wrapper.get_shipyards_id_opponent(),
            self.board_wrapper.get_shipyards_me_map(),
            self.board_wrapper.get_shipyards_opponent_map(),
            self.board_wrapper.get_max_spawn_me_map(),
            self.board_wrapper.get_max_spawn_opponent_map(),
            self.board_wrapper.get_cargo_me_map(),
            self.board_wrapper.get_cargo_opponent_map(),
            self.board_wrapper.get_feature_map_flight_plan_me(),
            self.board_wrapper.get_feature_map_flight_plan_opponent(),
            self.kore_map
        ])

        state['scalars'] = np.array([
            self.kore_me,
            self.kore_opponent,
            self.ship_count_me,
            self.ship_count_opponent,
            self.step_normalized,
            self.shipyard_count_me,
            self.shipyard_count_opponent
        ] + self.valid_action_mask)

        return state

    @staticmethod
    def get_input_shape() -> Dict[str, Union[Tuple[int, int, int], Tuple[int]]]:
        shapes = {
            'maps': (13, 21, 21),
            'scalars': (13,)
        }

        return shapes
