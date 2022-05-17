from kaggle_environments import make, Environment

from src.Agents.KoreAgent import KoreAgent
from src.Agents.SimpleAgent import SimpleAgent
from src.Monitoring.KoreMonitor import KoreMonitor


class KoreEnv:

    ENV_NAME: str = "kore_fleets"

    def __init__(self, agent: KoreAgent):
        #self.env: Environment = Environment(agents=[SimpleAgent(name="Dummy_agent")])
        self.env: Environment = make(self.ENV_NAME, debug=True)
        self.env.agents = [agent]

    def run(self):
        some_action = {'0-1': 'SPAWN_1'}
        while not self.env.done:
            self.env.step([some_action])

    def render_html(self) -> str:
        return self.env.render(mode="html", width=1000, height=800)
