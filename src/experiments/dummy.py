from src.Actions.action_adapter_rule_based import RuleBasedActionAdapter
from src.Actions.multi_action_adapter_rule_based import MultiActionAdapterRuleBased
from src.Agents.dqn_kore_agent import DQNKoreAgent
from src.Agents.neural_networks.mlp import get_mlp
from src.Environment.kore_env import KoreEnv
from src.Rewards.dummy_reward import DummyReward

from src.Rewards.survivor_reward import SurvivorReward
from src.States.advanced_state import AdvancedState


def main():
    reward = DummyReward()
    action_adapter = MultiActionAdapterRuleBased()

    kore_env = KoreEnv(AdvancedState, action_adapter, reward)
    model = get_mlp(AdvancedState.get_input_shape(), action_adapter.N_ACTIONS, window_length=2)
    kore_agent = DQNKoreAgent(kore_env=kore_env, model=model, training_steps=500000)
    kore_agent.fit()


if __name__ == "__main__":
    main()