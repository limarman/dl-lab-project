from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.a2c_agent import A2CAgent
from src.Agents.opponent_agents.handicapped_balanced_agent import HandicappedBalancedAgent
from src.Environment.kore_env_factory import KoreEnvFactory
from src.Monitoring.kore_monitor import KoreMonitor
from src.Rewards.win_reward import WinReward
from src.States.hybrid_state import HybridState


def main():
    state_constr = HybridState
    win_reward = WinReward()
    rule_based_action_adapter = ActionAdapterRuleBased()

    #handicapped_balanced = HandicappedBalancedAgent(1.0)
    #kore_env_factory = KoreEnvFactory(state_constr, rule_based_action_adapter, win_reward,
    #                                 enemy_agent=handicapped_balanced.balanced_agent)

    kore_env_factory = KoreEnvFactory(state_constr, rule_based_action_adapter, win_reward)
    env = kore_env_factory.build_multicore_env()

    kore_monitor = KoreMonitor()
    kore_monitor.set_run_name('Win Reward - Substeps')
    kore_agent = A2CAgent(env=env, kore_monitor=kore_monitor)
    kore_agent.fit()


if __name__ == "__main__":
    main()
