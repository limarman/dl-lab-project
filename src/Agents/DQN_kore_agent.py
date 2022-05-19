import gym
from kaggle_environments.envs.kore_fleets.helpers import Board
from keras import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy

from src.Agents.kore_agent import KoreAgent
from src.Environment.kore_env import KoreEnv
from src.States.board_wrapper import BoardWrapper


class DQNKoreAgent(KoreAgent):

    def __init__(self, name: str, kore_env: KoreEnv):
        super().__init__(name)

        self.kore_env = kore_env
        self.state_constr = kore_env.state_constr
        self.action_adapter = kore_env.action_adapter

        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + (445,)))
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.action_adapter.N_ACTIONS))
        self.model.add(Activation('linear'))

        memory = SequentialMemory(limit=50000, window_length=1)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                      value_min=.1, value_test=.05, nb_steps=40000)
        self.dqn = DQNAgent(model=self.model, nb_actions=self.action_adapter.N_ACTIONS,
                            memory=memory, nb_steps_warmup=2000, target_model_update=150,
                            policy=policy, train_interval=4, delta_clip=1.,
                            enable_double_dqn=True, enable_dueling_network=True)

        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def fit(self):
        self.dqn.fit(self.kore_env, nb_steps=100000, visualize=True, verbose=2)

    def step(self, obs, config):
        board = Board(obs, config)
        state = self.state_constr(board)
        agent_action = self.dqn.forward(state.tensor)
        board_wrapper = BoardWrapper(player_id=0, board=board)

        return self.action_adapter.agent_to_kore_action(agent_action, board_wrapper)
