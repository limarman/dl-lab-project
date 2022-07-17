from typing import List

import wandb
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Agents.a2c_agent import A2CAgent
from src.Environment.kore_env_factory import KoreEnvFactory
from src.Rewards.advantage_reward import AdvantageReward
from src.Rewards.win_reward import WinReward
from src.States.hybrid_state import HybridState


class WinRateEvaluator:

    def __init__(self, agent: A2CAgent, opponents: List[str], state_constr, wandb_run=None):
        self.__agent = agent
        self.__opponents = opponents
        self.__wandb_run = wandb_run
        self.__state_constr = state_constr

    def run(self):
        columns = ['Enemy Agent', 'Win rate mean', 'Win rate std']
        data = []
        for enemy_agent in self.__opponents:
            win_rate_mean, win_rate_std = self.__evaluate_against_enemy_agent(enemy_agent)
            data.append([enemy_agent, win_rate_mean, win_rate_std])

        if self.__wandb_run:
            table = wandb.Table(columns=columns, data=data)
            self.__wandb_run.log({'Evaluation against Enemies after Training': table})

    def __evaluate_against_enemy_agent(self, enemy_agent: str):
        env = self.__get_opponent_env(enemy_agent)

        win_rate_mean, win_rate_std = evaluate_policy(
            model=self.__agent.model,
            env=env,
            n_eval_episodes=10,
            deterministic=True,
        )

        print("-----------------------------------------------")
        print(f"Results against {enemy_agent}:")
        print(f"Win rate mean: {win_rate_mean}")
        print(f"Win rate std: {win_rate_std}")
        print("-----------------------------------------------")

        return win_rate_mean, win_rate_std

    def __get_opponent_env(self, enemy_agent: str) -> VecNormalize:
        advantage_reward = WinReward()
        rule_based_action_adapter = ActionAdapterRuleBased()

        kore_env_factory = KoreEnvFactory(
            state_constr=self.__state_constr,
            action_adapter=rule_based_action_adapter,
            kore_reward=advantage_reward,
            enemy_agent=enemy_agent,
        )

        return kore_env_factory.build_multicore_env()

