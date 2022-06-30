import numpy as np
from typing import Dict
from numpy import ndarray

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.States.board_wrapper import BoardWrapper


class SingleActionAdapterRuleBased(ActionAdapterRuleBased):
    """
    Processes a single action (intended for the substep kore_env)
    """

    def __init__(self):
        super().__init__()

    def agent_to_kore_action(self, agent_actions: ndarray, board_wrapper: BoardWrapper, shipyard) -> (Dict[str, str], Dict[str, str]):
        """
        Returns the action with highest probability
        """

        if np.sum(agent_actions) != 0:
            agent_actions_probs = agent_actions / np.sum(agent_actions)
            action_idx = np.random.choice(a=range(len(agent_actions)), p=agent_actions_probs)
        else:
            action_idx = np.random.choice(a=range(len(agent_actions)))

        action, name = self._get_rb_action(action_idx, shipyard, board_wrapper.board)
        kore_action = {shipyard.id: str(action)}
        kore_action_names = {shipyard.id: name}

        return kore_action, kore_action_names
