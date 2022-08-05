from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from src.Rewards.win_reward_for_eval import WinRewardForEval
from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Environment.kore_env_factory import KoreEnvFactory
from src.States.hybrid_state import HybridState

import os


def main():

    def evaluate_against_enemy_agent(enemy_agent: str, model_file_name, model_PPO):
        env = get_opponent_env(enemy_agent)
        if model_PPO:
            model = PPO.load(os.path.abspath(f"../models_for_evals/{model_file_name}"), env=env)
        else:
            model = A2C.load(f"../models_for_evals/{model_file_name}", env=env)

        win_rate_mean, win_rate_std = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=100,  # 100 or 200
            deterministic=False,
        )

        print("-----------------------------------------------")
        print(f"Results against {enemy_agent}:")
        print(f"Win rate mean: {win_rate_mean}")
        print(f"Win rate std: {win_rate_std}")
        print("-----------------------------------------------")

        return win_rate_mean, win_rate_std


    def get_opponent_env(enemy_agent: str):
        state_constr = HybridState

        win_reward = WinRewardForEval()
        rule_based_action_adapter = ActionAdapterRuleBased()

        kore_env_factory = KoreEnvFactory(
            state_constr=state_constr,
            action_adapter=rule_based_action_adapter,
            kore_reward=win_reward,
            enemy_agent=enemy_agent,
            selfplay=False,
            one_core = False
        )

        return kore_env_factory.build_multicore_env()


    opponents = ["balanced", "miner", "random"]
    model_file_names = os.listdir(os.path.abspath(f"../models_for_evals"))
    model_PPO = True  # set it false when evaluating A2C


    for model_file_name in model_file_names:

        print(f"Model file name: {model_file_name}")

        for opponent in opponents:

            evaluate_against_enemy_agent(opponent, model_file_name, model_PPO)


        print("\n")
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        print("\n\n")



if __name__ == '__main__':
    main()



