import multiprocessing

import gym

from typing import Union, Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecEnv, DummyVecEnv, VecNormalize

from src.Actions.action_adapter import ActionAdapter
from src.Environment.kore_env import KoreEnv
from src.Environment.selfplay_env import SelfPlayEnv
from src.Environment.reward_NN_train_env import RewardNNTrainEnv
from src.Environment.reward_NN_train_env_selfplay import RewardNNTrainEnvSelfplay
from src.Rewards.kore_reward import KoreReward


class KoreEnvFactory:

    def __init__(self, state_constr, action_adapter: ActionAdapter, kore_reward: KoreReward, run_id=None, enemy_agent: Union[str, Callable] = "balanced", selfplay=False, one_core=False, save_state=False):
        self.__state_constr = state_constr
        self.__action_adapter = action_adapter
        self.__kore_reward = kore_reward
        self.__run_id = run_id
        self.__enemy_agent = enemy_agent
        self.__selfplay= selfplay
        self.__one_core = one_core
        self.__save_state= save_state

    def build_multicore_env(self) -> VecNormalize:
        num_cpu_cores = multiprocessing.cpu_count()

        if self.__one_core:  # hack for testing -- remove later
            sub_proc_env = SubprocVecEnv([self.__build_monitor_env for _ in range(1)])
        else:
            sub_proc_env = SubprocVecEnv([self.__build_monitor_env for _ in
                                          range(num_cpu_cores)])  # builds n(=no of cores) threads for each environment

        sub_proc_norm_env = VecNormalize(sub_proc_env)  # stack multiple independent environment

        return sub_proc_norm_env

    def __build_monitor_env(self) -> Monitor:  # wrap core environment with monitor
        return Monitor(self.__build_kore_env())

    def __build_kore_env(self) -> gym.Env:  # build one kore environment
        if self.__selfplay:
            if self.__save_state:
                return RewardNNTrainEnvSelfplay(self.__state_constr, self.__action_adapter, self.__kore_reward, run_id=self.__run_id, enemy_agent=self.__enemy_agent)
            else:
                return SelfPlayEnv(self.__state_constr, self.__action_adapter, self.__kore_reward, enemy_agent=self.__enemy_agent, run_id=self.__run_id)
        else:
            if self.__save_state:
                return RewardNNTrainEnv(self.__state_constr, self.__action_adapter, self.__kore_reward, enemy_agent=self.__enemy_agent)
            else:
                return KoreEnv(self.__state_constr, self.__action_adapter, self.__kore_reward, enemy_agent=self.__enemy_agent)



