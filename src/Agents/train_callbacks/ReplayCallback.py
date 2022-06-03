import os

from rl.callbacks import Callback


class ReplayCallback(Callback):
    def __init__(self, step_func, interval=None, folder_name='default', enemy_agent='balanced'):
        super().__init__()
        self.step_func = step_func
        self.interval = interval
        self.folder_name = folder_name
        self.enemy_agent = enemy_agent

    def on_episode_end(self, episode, logs):
        if self.interval is not None and episode % self.interval == 0:
            self.__save_replay(episode)

    def __save_replay(self, episode):
        self.env.env.run([self.step_func, self.enemy_agent])
        replay_video = self.env.render()

        file_path = f"../../output/replays/{self.folder_name}"
        os.makedirs(file_path, exist_ok=True)
        with open(file_path + f"/replay_video_{episode}_episodes.html",
                  "w+", encoding='utf-8') as file:
            file.write(replay_video)
