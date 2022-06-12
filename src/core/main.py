from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.A2C_agent import A2CAgent
from src.Agents.dqn_kore_agent import DQNKoreAgent
from src.Agents.neural_networks.mlp import get_mlp
from src.Environment.kore_env import KoreEnv

from src.Rewards.dummy_reward import DummyReward
from src.Rewards.penalized_dummy_reward import PenalizedDummyReward
from src.States.advanced_state import AdvancedState


def main():
    dummy_reward = PenalizedDummyReward()
    action_adapter = ActionAdapterRuleBased()

    kore_env = KoreEnv(AdvancedState, action_adapter, dummy_reward)
    kore_agent = A2CAgent(kore_env=kore_env)
    kore_agent.fit()


if __name__ == "__main__":
    main()
