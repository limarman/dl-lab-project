import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.Agents.neural_networks.cnn_pytorch import BasicCNN


class HybridNet(BaseFeaturesExtractor):
    """
    Combines a CNN and MLP. These feature extractor are followed by another neural network.
    Adapted from https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(self, observation_space_dict: gym.spaces.Dict):
        super().__init__(observation_space_dict, features_dim=16+128)
        feature_extractors = {}
        output_len = 0

        for key, subspace in observation_space_dict.spaces.items():
            if key == "maps":
                feature_extractors[key] = BasicCNN(subspace.shape[0])
                output_len += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "scalars":
                # after the feature extractor is still the shared network
                # such that we use only a small network for fixed size input
                feature_extractors[key] = nn.Linear(subspace.shape[0], 16)
                output_len += 16
            else:
                raise Exception('Unknown key in environment dict')

        self.feature_extractors = nn.ModuleDict(feature_extractors)

    def forward(self, observations) -> torch.Tensor:
        extracted_features = []

        for key, extractor in self.feature_extractors.items():
            extracted_features.append(extractor(observations[key]))

        return torch.cat(extracted_features, dim=1)