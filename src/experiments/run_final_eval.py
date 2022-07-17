from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.a2c_agent import A2CAgent
from src.Environment.kore_env_factory import KoreEnvFactory
from src.Monitoring.kore_monitor import KoreMonitor
from src.Rewards.advantage_reward import AdvantageReward
from src.States.multimodal_state import MultimodalState
from src.experiments.win_rate_evaluator import WinRateEvaluator


def main():
    """
    Run the evaluation against opponents of a trained model
    """
    # run_id of model to evaluate
    run_id = 'local28b9e056-052a-11ed-a71a-acde48001122'
    state_constr = MultimodalState
    reward = AdvantageReward()
    feature_extractor = MultimodalState
    rule_based_action_adapter = ActionAdapterRuleBased()

    # load model and setup env
    kore_env_factory = KoreEnvFactory(state_constr, rule_based_action_adapter, reward)
    env = kore_env_factory.build_multicore_env()

    run_name = 'A2C: ' + reward.__class__.__name__ + ' - ' + feature_extractor.__name__
    kore_monitor = KoreMonitor(run_id, resume_training=True)
    kore_monitor.set_run_name(run_name)
    kore_agent = A2CAgent(env=env,
                          kore_monitor=kore_monitor,
                          n_training_steps=0,
                          resume_training=True,
                          run_id=run_id,
                          feature_extractor_class=feature_extractor)

    opponents = ["balanced", "random", "do_nothing", "miner"]
    win_rate_evaluator = WinRateEvaluator(kore_agent,
                                          opponents,
                                          state_constr,
                                          wandb_run=kore_monitor.run)
    win_rate_evaluator.run()


if __name__ == "__main__":
    main()
