import os
import wandb

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.a2c_agent import A2CAgent
from src.Environment.kore_env_factory import KoreEnvFactory
from src.Rewards.win_reward import WinReward
from src.States.hybrid_state import HybridState


def main():
    entity = os.environ.get("WANDB_ENTITY")
    wandb_run = wandb.init(
        project="rl-dl-lab-a2c",
        entity=entity,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    state_constr = HybridState
    win_reward = WinReward()
    rule_based_action_adapter = ActionAdapterRuleBased()

    kore_env_factory = KoreEnvFactory(state_constr, rule_based_action_adapter, win_reward)
    env = kore_env_factory.build_multicore_env()

    kore_agent = A2CAgent(env=env, wandb_run=wandb_run)
    kore_agent.fit()


if __name__ == "__main__":
    main()
