from rl.policy import Policy
import numpy as np


class BiasedSpawnyQPolicy(Policy):
    """Implements an dapted epsilon greedy policy that is biased towards
    spawning new ships

    The policy either:

    - takes a random action with probability epsilon/2
    - takes the spawn action with probability epsilon/2
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, eps=.5):
        super().__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        random_val = np.random.uniform()

        if random_val < (0.5 * self.eps):
            action = np.random.randint(0, nb_actions)
        elif random_val < self.eps:
            action = 0
        else:
            action = np.argmax(q_values)
        return action


