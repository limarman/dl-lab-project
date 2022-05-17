import os

from src.Actions.ActionAdapter import ActionAdapter
from src.Agents.DQNKoreAgent import DQNKoreAgent
from src.Environment.KoreEnv import KoreEnv

from src.Rewards.DummyReward import DummyReward
from src.States.DummyState import DummyState


def main():
    #kore_amount_monitor = KoreMonitor(agent_name=simple_agent.name, value_name="kore_amount")
    #simple_agent.register_monitor(kore_amount_monitor)

    dummy_reward = DummyReward()
    action_adapter = ActionAdapter()

    kore_env = KoreEnv(DummyState, action_adapter, dummy_reward)
    kore_agent = DQNKoreAgent(name="DQN_Kore_Agent", kore_env=kore_env)
    kore_agent.fit()

    kore_env.env.run([kore_agent.step])
    replay_video = kore_env.render()

    os.makedirs("../../output/replays", exist_ok=True)
    with open("../../output/replays/replay_video.html", "w+") as file:
        file.write(replay_video)


if __name__ == "__main__":
    main()
