import os

from rl.callbacks import Callback


class ReplayCallback(Callback):
    def __init__(self, step_func, interval=None):
        super().__init__()
        self.step_func = step_func
        self.interval = interval

    def on_episode_end(self, episode, logs):
        if self.interval is not None and episode % self.interval == 0:
            self.__save_replay(episode)

    def __save_replay(self, episode):
        self.env.env.run([self.step_func, 'balanced'])
        replay_video = self.env.render()

        os.makedirs("../../output/replays", exist_ok=True)
        with open(f"../../output/replays/replay_video_{episode}_episodes.html", "w+", encoding='utf-8') as file:
            file.write(replay_video)

