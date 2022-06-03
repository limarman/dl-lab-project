from kaggle_environments.envs.kore_fleets.helpers import Board
from keras import Model
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, Policy
from stable_baselines3 import A2C

from src.Agents.train_callbacks.ReplayCallback import ReplayCallback
from src.Agents.train_callbacks.WandbLogger import WandbLogger
from src.Environment.kore_env import KoreEnv
from src.States.board_wrapper import BoardWrapper


class A2CAgent:

    def __init__(
            self,
            kore_env: KoreEnv,
            training_steps: int = 1500000,
            name: str = 'dqg_agent',
    ):
        self.kore_env = kore_env
        self.state_constr = kore_env.state_constr
        self.action_adapter = kore_env.action_adapter
        self.training_steps = training_steps
        self.name = name

        self.model = A2C("MlpPolicy", kore_env, learning_rate=0.0004, verbose=1)

    def fit(self):
        callback = ReplayCallback(self.step, interval=500, folder_name=self.name)
        self.model.learn(total_timesteps=500000, callback=callback)

    def step(self, obs, config):
        board = Board(obs, config)
        state = self.state_constr(board)
        agent_action = self.model.predict(state.tensor)[0]
        board_wrapper = BoardWrapper(board=board, player_id=0)

        return self.action_adapter.agent_to_kore_action(agent_action, board_wrapper)
