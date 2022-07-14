import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.Agents.neural_networks.mlp_pytorch import BasicMLP


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
        embedding_dim = 64

        self.embedding = MultiModalEmbedding(patch_dim,
                                             patch_width,
                                             embedding_dim,
                                             num_patches,
                                             num_shipyards_scalars,
                                             num_game_stats)

        self.transformer = Transformer(self.HIDDEN_DIM, 8, 2, self.HIDDEN_DIM, self.HIDDEN_DIM, 0)

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
        shipyard_hidden_states = x[:,:num_shipyards]

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

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden_states = self.transformer(inputs['maps'], inputs['shipyards'], inputs['scalars'])
        # only use hidden state from the first shipyard
        # shape is (batch_dim, num_shipyards, hidden_state)
        return hidden_states[:,0,:]


class MultiModalEmbedding(nn.Module):

    def __init__(self,
                 patch_dim,
                 patch_width,
                 embedding_dim,
                 num_patches,
                 num_shipyard_scalars,
                 num_game_stats):
        super().__init__()

        self.maps_to_embeddings = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_width, s2=patch_width),
            BasicMLP(patch_dim, embedding_dim)
        )

        self.shipyards_to_embeddings = BasicMLP(num_shipyard_scalars, embedding_dim)
        self.game_stats_to_embedding = BasicMLP(num_game_stats, embedding_dim)

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
        # shape (batch_size, num_shipyards, embedding_size)
        embedded_patches = self.maps_to_embeddings(maps)
        # shape: (batch_size, num_patches, embedding_size)
        embedded_game_stats = embedded_game_stats.unsqueeze(1)

        # add position embedding
        embedded_patches += self.positions

        # TODO maybe add sinusoidal position encodings

        # start with shipyards (order is important since we need to retrieve
        # the classification tensors in the same order)

        return torch.cat([embedded_shipyards,
                          embedded_game_stats,
                          embedded_patches], dim=1)

"""
Following code has been adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
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

# TODO remove
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

batch_size = 2
game_stats_num = 15
shipyard_stats = 5
max_shipyards = 15

feature_maps = torch.rand((batch_size, 15, 21, 21))
shipyards = torch.randn((batch_size, max_shipyards, shipyard_stats))
game_stats = torch.randn((batch_size, game_stats_num))

input_dict = {
            'maps': feature_maps,
            'scalars': game_stats,
            'shipyards': shipyards
        }

spaces = {
    'maps': gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(15, 21, 21)),
    'scalars': gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(15,)),
    'shipyards': gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(15,5))
}

spaces_dict = gym.spaces.Dict(spaces)

model = MultiModalNet(spaces_dict)
print(model(input_dict).shape)

