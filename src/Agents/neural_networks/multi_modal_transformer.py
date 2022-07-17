from typing import Dict

import math
import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.Agents.neural_networks.mlp_pytorch import BasicMLP, TinyMLP


class MultiModalTransformer(nn.Module):
    """
    Embeds the shipyard information, game information and feature maps
    into a vector respresentation that is then fed into an transformere encoder.
    Finally, a classifier head is applied for each shipyard or the shipyards
    hidden representation is returned (if apply_mlp_head=False).

    The latter case is beneficial since the MultiModalTransformer is shared between
    policy and value network whereas the mlp_head is individual for each of them.

    Model input:
    - feature maps: (batch_size, channel, width, height)
    - shipyards: (batch_size, max_num_shipyards, num_shipyards_scalars)
    - game_stats: (batch_size, num_game_stats)

    Model output:
    if apply_mlp_head:
        - Predictions for shipyards: (batch_size, max_num_shipyards, logits)
    else:
        - hidden state for shipyards: (batch_size, max_num_shipyards, hidden_dim)
    """

    HIDDEN_DIM = 64

    def __init__(self,
                 num_channels,
                 num_shipyards_scalars,
                 num_game_stats,
                 num_actions=None,
                 apply_mlp_head=False):
        super().__init__()
        self.apply_mlp_head = apply_mlp_head
        num_patches = 7 * 7
        patch_dim = 3 * 3 * num_channels
        patch_width = 3
        embedding_dim = self.HIDDEN_DIM

        self.embedding = MultiModalEmbedding(patch_dim,
                                             patch_width,
                                             embedding_dim,
                                             num_patches,
                                             num_shipyards_scalars,
                                             num_game_stats)

        self.transformer = Transformer(self.HIDDEN_DIM, 4, 4, self.HIDDEN_DIM, self.HIDDEN_DIM, 0)

        if apply_mlp_head:
            if not num_actions:
                raise Exception('Number of actions must be given to apply MLP head')
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(64),
                nn.Linear(64, num_actions)
            )

    def forward(self, maps, shipyards, game_stats):
        x = self.embedding(maps, shipyards, game_stats)
        x = self.transformer(x)

        _, num_shipyards, _ = shipyards.shape
        shipyard_hidden_states = x[:, :num_shipyards]

        if self.apply_mlp_head:
            out = self.classifier_head(shipyard_hidden_states)
        else:
            out = shipyard_hidden_states

        return out


class MultiModalNet(BaseFeaturesExtractor):
    """
    Wraps the multimodal transformer inside a stablebaseline3 feature exctractor class.
    The feature extractor is followed  by another neural network that functions as
    the MLP head (the feature extractor is hence shared by policy
    and value network, whereas the MLP head is different for each of them)

    In this implementation, we only consider the output for one shipyard
    due to the advantages of the substep method (updating kore, ships etc.
    before choosing the next action)

    Adapted from https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

    Returns the hidden state for the first shipyard (i.e. (batch_size, max_num_shipyards, hidden_dim))
    """

    def __init__(self, obs_space_dict: gym.spaces.Dict):
        super().__init__(obs_space_dict, features_dim=MultiModalTransformer.HIDDEN_DIM)
        # batch dim is not contained in the shapes
        num_channels = obs_space_dict['maps'].shape[0]
        num_shipyard_scalars = obs_space_dict['shipyards'].shape[1]
        num_game_scalars = obs_space_dict['scalars'].shape[0]
        # fixed shipyard size is required for batch training, however we could apply masks
        # or take  the max_num of shipyards in the batch as padding len
        self.transformer = MultiModalTransformer(num_channels, num_shipyard_scalars, num_game_scalars)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden_states = self.transformer(inputs['maps'], inputs['shipyards'], inputs['scalars'])
        # only use hidden state from the first shipyard
        # shape is (batch_dim, num_shipyards, hidden_state)
        return hidden_states[:, 0, :]


class MultiModalEmbedding(nn.Module):

    def __init__(self,
                 patch_dim,
                 patch_width,
                 embedding_dim,
                 num_patches,
                 num_shipyard_scalars,
                 num_game_stats,
                 sinusoidal_embedding=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sinusoidal_emebedding = sinusoidal_embedding

        self.maps_to_patches = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_width, s2=patch_width)
        self.maps_to_embeddings = nn.Sequential(
            self.maps_to_patches,
            TinyMLP(patch_dim, embedding_dim)
        )

        self.shipyards_to_embeddings = TinyMLP(num_shipyard_scalars, embedding_dim)
        self.game_stats_to_embedding = TinyMLP(num_game_stats, embedding_dim)

        if sinusoidal_embedding:
            self.positions = self._get_positional_encoding_for_patches()
        else:
            self.positions = nn.Parameter(torch.randn(num_patches, embedding_dim))

    def forward(self, maps, shipyards, game_stats):
        """
        Returns (batch_size, max_shipyards + game_stats + num_patches, embedding dim)
                = (batch_size, max_shipyards + 1, embedding_dime)
        """
        batch_size = maps.size(0)

        embedded_shipyards = self.shipyards_to_embeddings(shipyards)
        # shape: (batch_size, num_shipyards, embedding_size)
        embedded_game_stats = self.game_stats_to_embedding(game_stats)
        # shape (batch_size, embedding_size)
        embedded_patches = self.maps_to_embeddings(maps)
        # shape: (batch_size, num_patches, embedding_size)
        embedded_game_stats = embedded_game_stats.unsqueeze(1)

        # add position embedding
        embedded_patches += self.positions

        if self.sinusoidal_emebedding:
            embedded_shipyards += self._get_positional_encoding_for_shipyards(shipyards)

        # start with shipyards (order is important since we need to retrieve
        # the classification tensors in the same order)

        return torch.cat([embedded_shipyards,
                          embedded_game_stats,
                          embedded_patches], dim=1)

    def _get_positional_encoding(self, x: int, y: int) -> torch.Tensor:
        """
        We used the 2D-Adaption of sinusoidal positional embedding of
        Raisi, Zobeir, et al. "2D positional embedding-based transformer for scene text recognition."

        This means to encode the x_coordinate in the upper embedding half and the y_coordinate in
        the lower embedding half
        """

        assert self.embedding_dim % 4 == 0, "embedding_dim must be dividable by four"

        encoding = torch.zeros((self.embedding_dim))

        for i in range(self.embedding_dim // 4):
            d_half = self.embedding_dim // 2
            factor = (1e-4) ** (4 * i / self.embedding_dim)

            encoding[2 * i] = math.sin(x * factor)
            encoding[2 * i + 1] = math.cos(x * factor)
            encoding[2 * i + d_half] = math.sin(y * factor)
            encoding[2 * i + 1 + d_half] = math.cos(y * factor)

        return encoding

    def _get_positional_encoding_for_patches(self) -> torch.Tensor:
        """
        Calculates the positional embedding for patches. We take the position in the middle as reference.
        We simulate the patch decomposition with natural numbers to make sure, that we have the correct positions.

        Returns encodings of shape: (batch_size, num_patches, embedding_dim)
        """
        numbers = torch.arange(0, 21)
        y_map = torch.stack([numbers]*21)
        x_map = torch.transpose(y_map,0,1)

        # adding batch and channel dim
        y_map = y_map.view(1,1,21,21)
        x_map = x_map.view(1,1,21,21)
        y_patches = self.maps_to_patches(y_map)
        x_patches = self.maps_to_patches(x_map)

        num_patches = y_patches.size(1)
        encoding = []
        for i in range(num_patches):
            # x_patches have the form (n,n,n,n+,n+1, ...)
            x_middle = x_patches[0,i,4]
            # y_patches have the form (n, n+1, n+2, n, n+1, ...)
            y_middle = y_patches[0,i,1]
            encoding.append(self._get_positional_encoding(x_middle,y_middle))

        return torch.stack(encoding)

    def _get_positional_encoding_for_shipyards(self, shipyards) -> torch.Tensor:
        """
        Calculates the positional encoding for shipyards with
        shape (batch_size, num_shipyards, ebedding_dim)

        The input shipyard has shape (batch_size, num_shipyards, num_shipyard_features)
        """
        encodings = []

        num_shipyards = shipyards.size(1)

        for i in range(num_shipyards):
            x = shipyards[0, i, 0]
            y = shipyards[0, i, 1]
            encoding = self._get_positional_encoding(x, y)
            encodings.append(encoding)

        return torch.stack(encodings)


"""
***********************************************************************************************************
Following code has been adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
***********************************************************************************************************
"""


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
