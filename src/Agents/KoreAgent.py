from abc import ABC, abstractmethod
from typing import List

from src.Monitoring.KoreMonitor import KoreMonitor


class KoreAgent(ABC):

    def __init__(self, name: str):
        self.name = name
        self.monitors: List[KoreMonitor] = []

    def register_monitor(self, monitor: KoreMonitor):
        self.monitors.append(monitor)

    @abstractmethod
    def step(self, obs, config):
        pass
