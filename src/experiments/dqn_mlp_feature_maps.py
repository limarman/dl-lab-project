import os

from src.Actions.action_adapter import ActionAdapter
from src.Agents.DQN_kore_agent import DQNKoreAgent
from src.Environment.kore_env import KoreEnv

from src.Rewards.simple_reward import SimpleReward
from src.States.simple_state import SimpleState




def main():
    simple_reward = SimpleReward()
    action_adapter = ActionAdapter()

    kore_env = KoreEnv(SimpleState, action_adapter, simple_reward)
    kore_agent = DQNKoreAgent(name="DQN_Kore_Agent", kore_env=kore_env, input_size=SimpleState.get_input_shape(), training_steps=1000000)
    kore_agent.fit()

    kore_env.env.run([kore_agent.step, 'balanced'])
    replay_video = kore_env.render()

    os.makedirs("../../output/replays", exist_ok=True)
    with open("../../output/replays/replay_video.html", "w+", encoding='utf-8') as file:
        file.write(replay_video)


if __name__ == "__main__":
    main()