import itertools
from typing import Dict

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.States.board_wrapper import BoardWrapper


class MultiActionAdapterRuleBased(ActionAdapterRuleBased):
    """
    Adapts the rule based action adapter such that for each shipyard an action is produced.
    The action space is approximated by actions^max_num_shipyards which only works for a
    low number of shipyards
    """

    MAX_NUM_SHIPYARDS = 4
    NUM_ACTIONS_PER_SHIPYARD = 4
    N_ACTIONS: int = NUM_ACTIONS_PER_SHIPYARD ** MAX_NUM_SHIPYARDS # expand, attack, farm, build


    def __init__(self):
        super().__init__()
        self.action_space = list(itertools.product(*[list(range(self.NUM_ACTIONS_PER_SHIPYARD))] * self.NUM_ACTIONS_PER_SHIPYARD))

    def agent_to_kore_action(self, agent_action: int, board_wrapper: BoardWrapper) -> Dict[str, str]:
        action_sequence = self.action_space[agent_action]
        shipyards = board_wrapper.get_shipyards_of_current_player()
        shipyards = sorted(shipyards, key=lambda x: x.id)

        actions = {}
        for (shipyard, action_idx) in zip(shipyards, action_sequence):
            # zip stops when the end of the shorter list is reached
            action = self._get_rb_action(action_idx, shipyard, board_wrapper.board)
            actions[shipyard.id] = str(action)
        return actions

