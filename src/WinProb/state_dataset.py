import os
import torch
from torch.utils.data import Dataset


class StateDataset(Dataset):
    """
    The Class will act as the container for state dataset.
    Each state is made up of maps and scalars.
    This will also output the ID of the winning agent.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.win_name_list = os.listdir(os.path.join(os.path.abspath(self.root_dir), 'win'))
        self.lose_name_list = os.listdir(os.path.join(os.path.abspath(self.root_dir), 'lose'))
        self.win_length = len(self.win_name_list)
        self.lose_length = len(self.lose_name_list)

    def __len__(self):
        # Return the number of batches
        return self.win_length + self.lose_length

    def __getitem__(self, idx):
        # Return a full batch based on an index. Ex. dataset[0] will return the first batch from the dataset, in this case the states and the wining agents.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < self.win_length:
            state_name = self.win_name_list[idx]
            state = torch.load(os.path.join(os.path.abspath(self.root_dir), 'win', state_name))
            label = 0
        else:
            loc = idx - self.win_length
            state_name = self.lose_name_list[loc]
            state = torch.load(os.path.join(os.path.abspath(self.root_dir), 'lose', state_name))
            label = 1

        if self.transform:
            state['maps'] = self.transform(state['maps'])

        return torch.tensor(state['maps']).float(), torch.tensor(state['scalars']).float(), label