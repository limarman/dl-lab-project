from src.Actions.action_adapter_rule_based import RuleBasedActionAdapter
from src.Agents.a2c_agent import A2CAgent
from src.Environment.kore_env_factory import KoreEnvFactory
from src.Rewards.advantage_reward import AdvantageReward
from src.States.advanced_state import AdvancedState


def main():
    state_constr = AdvancedState
    advantage_reward = AdvantageReward()
    rule_based_action_adapter = RuleBasedActionAdapter()

    kore_env_factory = KoreEnvFactory(state_constr, rule_based_action_adapter, advantage_reward)
    kore_agent = A2CAgent(env=kore_env_factory.build_multicore_env())
    kore_agent.fit()


if __name__ == "__main__":
    main()
