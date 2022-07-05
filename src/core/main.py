import os
import uuid

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.a2c_agent import A2CAgent
from src.Agents.neural_networks.hybrid_net import HybridResNet, HybridTransformer, HybridNetBasicCNN
from src.Environment.kore_env_factory import KoreEnvFactory
from src.Monitoring.kore_monitor import KoreMonitor
from src.Rewards.advantage_reward import AdvantageReward
from src.Rewards.win_reward import WinReward
from src.States.hybrid_state import HybridState


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
        n_training_steps = 8500000

    state_constr = HybridState
    win_reward = AdvantageReward()
    rule_based_action_adapter = ActionAdapterRuleBased()

    kore_env_factory = KoreEnvFactory(state_constr, rule_based_action_adapter, win_reward)
    env = kore_env_factory.build_multicore_env()

    if os.path.exists(f"checkpoints/{run_id}.zip"):
        resume_training = True
    else:
        resume_training = False

    kore_monitor = KoreMonitor(run_id, resume_training=resume_training)
    kore_monitor.set_run_name('Win Reward - Substeps')
    kore_agent = A2CAgent(env=env,
                          kore_monitor=kore_monitor,
                          n_training_steps=n_training_steps,
                          resume_training=resume_training,
                          run_id=run_id,
                          feature_extractor_class=HybridNetBasicCNN)
    kore_agent.fit()


if __name__ == "__main__":
    main()
