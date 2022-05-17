from src.Agents.KoreAgent import KoreAgent

from kaggle_environments.envs.kore_fleets.helpers import *


class SimpleAgent(KoreAgent):

    def __init__(self, name: str = "SimpleAgent"):
        super().__init__(name)

    def step(self, obs, config):
        board = Board(obs, config)
        me = board.current_player
        spawn_cost = board.configuration.spawn_cost
        kore_left = me.kore

        for shipyard in me.shipyards:
            if kore_left >= spawn_cost:
                action = ShipyardAction.spawn_ships(1)
                shipyard.next_action = action
            elif shipyard.ship_count > 0:
                direction = Direction.NORTH
                action = ShipyardAction.launch_fleet_with_flight_plan(2, direction.to_char())
                shipyard.next_action = action

        self.monitors[0].log_value(kore_left)

        print(me.next_actions)
        return me.next_actions
