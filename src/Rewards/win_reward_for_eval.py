from src.Rewards.win_reward import WinReward


class WinRewardForEval(WinReward):
    """
    Implementation for the 0-1-win reward to calculate correct win rate
    Our other win reward is -1 if we loose and +1 if we win
    """

    def __init__(self):
        super().__init__()

    def get_reward(self, *args):
        reward = super().get_reward(*args)
        non_negative_reward = max(0, reward)
        return non_negative_reward
