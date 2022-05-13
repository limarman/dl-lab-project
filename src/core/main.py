from src.Agents.SimpleAgent import SimpleAgent
from src.Environment.KoreEnv import KoreEnv
from src.Monitoring.KoreMonitor import KoreMonitor


def main():
    replay_video = ""

    with open("../../output/replays/replay_video.html", "w") as file:
        file.write(replay_video)


if __name__ == "__main__":
    main()
