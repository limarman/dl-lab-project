import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.Agents.neural_networks.cnn_pytorch import BasicCNN
from src.Agents.neural_networks.mlp_pytorch import BasicMLP


class HybridNet(BaseFeaturesExtractor):
    """
    Combines a CNN and MLP. These feature extractor are followed by another neural network.
    Adapted from https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(self, observation_space_dict: gym.spaces.Dict):
        cnn_output_features = 128
        mlp_output_features = 64
        output_features = cnn_output_features + mlp_output_features
        super().__init__(observation_space_dict, features_dim=output_features)
        feature_extractors = {}

        for key, subspace in observation_space_dict.spaces.items():
            if key == "maps":
                feature_extractors[key] = BasicCNN(subspace.shape[0], cnn_output_features)
            elif key == "scalars":
                feature_extractors[key] = BasicMLP(subspace.shape[0], mlp_output_features)
            else:
                raise Exception('Unknown key in environment dict')

        self.feature_extractors = nn.ModuleDict(feature_extractors)

    def forward(self, observations) -> torch.Tensor:
        extracted_features = []

        for key, extractor in self.feature_extractors.items():
            extracted_features.append(extractor(observations[key]))

        return torch.cat(extracted_features, dim=1)
