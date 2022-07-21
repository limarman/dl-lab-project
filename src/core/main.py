import os
import signal
import uuid

from stable_baselines3.common.evaluation import evaluate_policy

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.a2c_agent import A2CAgent
from src.Agents.neural_networks.hybrid_net import HybridResNet, HybridTransformer, HybridNetBasicCNN
from src.Agents.neural_networks.multi_modal_transformer import MultiModalNet
from src.Environment.kore_env_factory import KoreEnvFactory
from src.Monitoring.kore_monitor import KoreMonitor
from src.Rewards.advantage_reward import AdvantageReward
from src.Rewards.competitive_kore_delta_reward import CompetitiveKoreDeltaReward
from src.Rewards.win_reward import WinReward
from src.States.hybrid_state import HybridState
from src.experiments.win_rate_evaluator import WinRateEvaluator
from src.States.multimodal_state import MultimodalState



def main():
    """
    Supports chain jobs on the cluster and local execution
    """
    if os.environ.get('SLURM_ARRAY_JOB_ID'):
        # use the unique cluster job id
        run_id = 'cluster' + os.environ.get('SLURM_ARRAY_JOB_ID')
        # run until the cluster kills us
        n_training_steps = 150000000
    else:
        run_id = 'local' + str(uuid.uuid1())
        n_training_steps = 5000000

    state_constr = HybridState
    feature_extractor = HybridResNet
    reward = AdvantageReward()
    #reward = CompetitiveKoreDeltaReward()
    rule_based_action_adapter = ActionAdapterRuleBased()

    kore_env_factory = KoreEnvFactory(state_constr, rule_based_action_adapter, reward)
    env = kore_env_factory.build_multicore_env()

    if os.path.exists(f"checkpoints/{run_id}.zip"):
        resume_training = True
    else:
        resume_training = False

    run_name = 'A2C: ' + reward.__class__.__name__ + ' - ' + feature_extractor.__name__
    kore_monitor = KoreMonitor(run_id, resume_training=resume_training)
    kore_monitor.set_run_name(run_name)
    kore_agent = A2CAgent(env=env,
                          kore_monitor=kore_monitor,
                          n_training_steps=n_training_steps,
                          resume_training=resume_training,
                          run_id=run_id,
                          feature_extractor_class=feature_extractor)

    opponents = ["balanced", "random", "do_nothing", "miner"]


    def handle_interrupt(sigmun, frame):
        # save model and evaluate after training
        # time for interrupt can be specified in the jobscript
        print('Catched Interrupt')
        win_rate_evaluator = WinRateEvaluator(kore_agent,
                                              opponents,
                                              state_constr,
                                              wandb_run=kore_monitor.run)
        kore_agent.model.save(f"checkpoints/{run_id}")
        print('Model saved')
        win_rate_evaluator.run()
        print('Final evaluation is completed')

    signal.signal(signal.SIGTERM, handle_interrupt)

    kore_agent.fit()

if __name__ == "__main__":
    main()
