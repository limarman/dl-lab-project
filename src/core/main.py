from kaggle_environments import make


def main():
    env = make("kore_fleets", debug=True)
    print(env.name, env.version)

    env.run(["balanced"])
    replay_video = env.render(mode="html", width=1000, height=800)
    print(replay_video)

    with open("../../output/replays/replay_video.html", "w") as file:
        file.write(replay_video)


if __name__ == "__main__":
    main()
