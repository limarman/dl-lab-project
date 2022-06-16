import multiprocessing

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.Actions.action_adapter import ActionAdapter
from src.Environment.kore_env import KoreEnv
from src.Rewards.kore_reward import KoreReward


class KoreEnvFactory:

    def __init__(self, state_constr, action_adapter: ActionAdapter, kore_reward: KoreReward):
        self.__state_constr = state_constr
        self.__action_adapter = action_adapter
        self.__kore_reward = kore_reward

    def build_multicore_env(self) -> SubprocVecEnv:
        num_cpu_cores = multiprocessing.cpu_count()
        return SubprocVecEnv([self.__build_monitor_env for _ in range(num_cpu_cores)])

    def __build_monitor_env(self) -> Monitor:
        return Monitor(self.__build_kore_env())

    def __build_kore_env(self) -> gym.Env:
        return KoreEnv(self.__state_constr, self.__action_adapter, self.__kore_reward)
