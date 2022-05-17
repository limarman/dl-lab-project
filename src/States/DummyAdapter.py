import numpy as np

from src.States.KoreState import KoreState
from src.States.StateAdapter import StateAdapter


class DummyAdapter(StateAdapter):

    def to_state(self, obs, config) -> KoreState:
        return KoreState(values=np.array([0, 0]))

