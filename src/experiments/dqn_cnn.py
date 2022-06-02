from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.baselines.random_rule_based import agent_without_expand
from src.Agents.dqn_kore_agent import DQNKoreAgent
from src.Agents.neural_networks.cnn import get_cnn
from src.Environment.kore_env import KoreEnv
from src.Rewards.advanced_reward import AdvancedReward
from src.Rewards.dummy_reward import DummyReward
from src.Rewards.penalized_dummy_reward import PenalizedDummyReward
from src.States.map_state import MapState


def cnn_exp(reward):
    action_adapter = ActionAdapterRuleBased(single_shipyard=True)

    kore_env = KoreEnv(MapState, action_adapter, reward, enemy_agent=agent_without_expand)
    model = get_cnn(MapState.get_input_shape(), action_adapter.N_ACTIONS, 1)
    reward_string = reward.__class__.__name__
    kore_agent = DQNKoreAgent(kore_env=kore_env, model=model, training_steps=250000, name='CNN: Playing against random single - '+reward_string)
    kore_agent.fit()


if __name__ == "__main__":
    for reward in [PenalizedDummyReward(), DummyReward(), AdvancedReward()]:
        cnn_exp(reward)
