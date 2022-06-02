from src.Actions.action_adapter import ActionAdapter
from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.dqn_kore_agent import DQNKoreAgent
from src.Agents.neural_networks.cnn import get_cnn
from src.Environment.kore_env import KoreEnv
from src.Rewards.dummy_reward import DummyReward
from src.Rewards.penalized_dummy_reward import PenalizedDummyReward
from src.States.dummy_state_map import DummyStateMap
from src.States.map_state import MapState


def main():
    #kore_amount_monitor = KoreMonitor(agent_name=simple_agent.name, value_name="kore_amount")
    #simple_agent.register_monitor(kore_amount_monitor)

    dummy_reward = PenalizedDummyReward()
    action_adapter = ActionAdapterRuleBased()

    kore_env = KoreEnv(MapState, action_adapter, dummy_reward)
    model = get_cnn(MapState.get_input_shape(), action_adapter.N_ACTIONS, 1)
    kore_agent = DQNKoreAgent(kore_env=kore_env, model=model)
    kore_agent.fit()


if __name__ == "__main__":
    main()
