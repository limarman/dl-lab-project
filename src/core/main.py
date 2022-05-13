from src.Agents.SimpleAgent import SimpleAgent
from src.Environment.KoreEnv import KoreEnv
from src.Monitoring.KoreMonitor import KoreMonitor


def main():
    simple_agent = SimpleAgent()

    kore_amount_monitor = KoreMonitor(agent_name=simple_agent.name, value_name="kore_amount")
    simple_agent.register_monitor(kore_amount_monitor)

    kore_env = KoreEnv()
    kore_env.run_agent(simple_agent)

    replay_video = kore_env.render_html()

    with open("../../output/replays/replay_video.html", "w") as file:
        file.write(replay_video)


if __name__ == "__main__":
    main()
