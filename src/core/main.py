from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.dqn_kore_agent import DQNKoreAgent
from src.Agents.neural_networks.mlp import get_mlp
from src.Environment.kore_env import KoreEnv

from src.Rewards.dummy_reward import DummyReward
from src.States.simple_state import SimpleState


def main():
    dummy_reward = DummyReward()
    action_adapter = ActionAdapterRuleBased()

    kore_env = KoreEnv(SimpleState, action_adapter, dummy_reward)
    model = get_mlp(SimpleState.get_input_shape(), action_adapter.N_ACTIONS, window_length=4)
    kore_agent = DQNKoreAgent(kore_env=kore_env, model=model)
    kore_agent.fit()


if __name__ == "__main__":
    main()
