import numpy as np
from kaggle_environments.envs.kore_fleets.helpers import *

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Actions.action_validity_masking import get_action_validity_mask
from src.States.board_wrapper import BoardWrapper
from src.States.kore_state import KoreState


class MultimodalState(KoreState):
    """
    A state combining map-shaped input and scalar input that can generate input for the MultiModalNet
    """

    def __init__(self, board: Board, shipyard: Shipyard = None):
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
        self.current_shipyard = shipyard
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
        super(MultimodalState, self).__init__(self.get_input_shape(), tensor, self.board_wrapper)

    def _get_tensor(self) -> Dict[str, np.ndarray]:
        """
        Returns a dict with a map-shaped output for keys
        'maps': (num_feature_maps, 21, 21)
        'scalars': (num_scalars,)
        'shipyards': (max_num_shipyards, num_shipyard_scalars)
        """
        state = {}

        state['maps'] = self.board_wrapper.get_feature_map_collection()

        # TODO remove shipyard scalars here?
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

        # apply zero padding for efficient batch training
        shipyard_vectors = np.zeros((15, 10))
        shipyard_infos = self.board_wrapper.get_shipyard_vectors(start_with=self.current_shipyard)[:15]
        if shipyard_infos.any():
            # if there really are shipyards
            shipyard_vectors[:shipyard_infos.shape[0], :shipyard_infos.shape[1]] = shipyard_infos

        state['shipyards'] = shipyard_vectors

        return state

    @staticmethod
    def get_input_shape() -> Dict[str, Union[Tuple[int, int, int], Tuple[int]]]:
        shapes = {
            'maps': (15, 21, 21),
            'scalars': (17,),
            'shipyards': (15, 10)
        }

        return shapes
