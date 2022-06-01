from kaggle_environments.envs.kore_fleets.helpers import Board
from keras import Model
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, Policy
from src.Agents.train_callbacks.ReplayCallback import ReplayCallback
from src.Agents.train_callbacks.WandbLogger import WandbLogger
from src.Environment.kore_env import KoreEnv
from src.States.board_wrapper import BoardWrapper


class DQNKoreAgent:

    def __init__(
            self,
            kore_env: KoreEnv,
            model: Model,
            training_steps: int = 150000,
            qpolicy: Policy = EpsGreedyQPolicy(),
            name: str = 'dqg_agent',
    ):
        self.kore_env = kore_env
        self.state_constr = kore_env.state_constr
        self.action_adapter = kore_env.action_adapter
        self.training_steps = training_steps
        self.name = name

        self.model = model
        self.window_length = model.input_shape[1]

        memory = SequentialMemory(limit=1000000, window_length=self.window_length)
        policy = LinearAnnealedPolicy(qpolicy, attr='eps', value_max=1.,
                                      value_min=.1, value_test=.05, nb_steps=100000)

        self.dqn = DQNAgent(model=self.model, nb_actions=self.action_adapter.N_ACTIONS,
                            memory=memory, nb_steps_warmup=100, target_model_update=10000,
                            policy=policy, train_interval=4, delta_clip=1., gamma=1,
                            enable_double_dqn=True, enable_dueling_network=True, batch_size=32)

        self.dqn.compile(Adam(lr=0.0001), metrics=['mae'])

    def fit(self):
        wandb_logger = WandbLogger(name=self.name)
        callbacks = [ReplayCallback(self.step, interval=20, folder_name=self.name,
                                    enemy_agent=self.kore_env.opponent_agent),
                     wandb_logger]
        self.dqn.fit(self.kore_env, nb_steps=self.training_steps, visualize=True, verbose=2, callbacks=callbacks)

    def step(self, obs, config):
        board = Board(obs, config)
        state = self.state_constr(board)
        agent_action = self.dqn.forward(state.tensor)
        board_wrapper = BoardWrapper(board=board, player_id=0)

        return self.action_adapter.agent_to_kore_action(agent_action, board_wrapper)
