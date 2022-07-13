from stable_baselines3.common.evaluation import evaluate_policy

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.a2c_agent import A2CAgent
from src.Environment.kore_env_factory import KoreEnvFactory
from src.Rewards.win_reward import WinReward
from src.States.hybrid_state import HybridState


class WinRateEvaluator:

    def __init__(self, agent: A2CAgent):
        state_constr = HybridState
        advantage_reward = WinReward()
        rule_based_action_adapter = ActionAdapterRuleBased()

        kore_env_factory = KoreEnvFactory(state_constr, rule_based_action_adapter, advantage_reward)
        env = kore_env_factory.build_multicore_env()

        win_rate_mean, win_rate_std = evaluate_policy(
            model=agent.model,
            env=env,
            n_eval_episodes=2,
            deterministic=True,
        )

        print(f"Win rate mean: {win_rate_mean}")
        print(f"Win rate std: {win_rate_std}")
