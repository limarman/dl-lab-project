from src.Actions.action_adapter import ActionAdapter
from src.Agents.DQN_kore_agent import DQNKoreAgent
from src.Agents.neural_networks.mlp import get_mlp
from src.Agents.policies.spawn_policy import BiasedSpawnyQPolicy
from src.Environment.kore_env import KoreEnv

from src.Rewards.dummy_reward import DummyReward
from src.States.dummy_state import DummyState


def main():
    #kore_amount_monitor = KoreMonitor(agent_name=simple_agent.name, value_name="kore_amount")
    #simple_agent.register_monitor(kore_amount_monitor)

    dummy_reward = DummyReward()
    action_adapter = ActionAdapter()

    kore_env = KoreEnv(DummyState, action_adapter, dummy_reward)
    model = get_mlp(DummyState.get_input_shape(), action_adapter.N_ACTIONS, train_interval=4)
    kore_agent = DQNKoreAgent(name="DQN_Kore_Agent", kore_env=kore_env, model=model, qpolicy=BiasedSpawnyQPolicy())
    kore_agent.fit()


if __name__ == "__main__":
    main()
