from typing import List

from kaggle_environments import make, Environment

from src.Agents.KoreAgent import KoreAgent
from src.Monitoring.KoreMonitor import KoreMonitor


class KoreEnv:

    ENV_NAME: str = "kore_fleets"

    def __init__(self):
        self.env: Environment = make(self.ENV_NAME, debug=True)

    def run_agent(self, agent: KoreAgent):
        self.env.run([agent.step])

    def render_html(self) -> str:
        return self.env.render(mode="html", width=1000, height=800)
