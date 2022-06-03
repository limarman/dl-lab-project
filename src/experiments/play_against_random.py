from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Actions.multi_action_adapter_rule_based import MultiActionAdapterRuleBased
from src.Agents.baselines.random_rule_based import agent_with_expand
from src.Agents.dqn_kore_agent import DQNKoreAgent
from src.Agents.neural_networks.mlp import get_mlp
from src.Environment.kore_env import KoreEnv
from src.Rewards.dummy_reward import DummyReward

from src.Rewards.survivor_reward import SurvivorReward
from src.States.advanced_state import AdvancedState

"""
Test our agent against randomly applying rule-based actor to check if we learn anything
"""


def main():
    reward = DummyReward()
    action_adapter = MultiActionAdapterRuleBased()

    kore_env = KoreEnv(AdvancedState, action_adapter, reward, enemy_agent=agent_with_expand)
    model = get_mlp(AdvancedState.get_input_shape(), action_adapter.N_ACTIONS, window_length=2)
    kore_agent = DQNKoreAgent(kore_env=kore_env, model=model, training_steps=200000, name='Playing against random - dummy reward + advanced state')
    kore_agent.fit()


if __name__ == "__main__":
    main()