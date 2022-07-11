import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

"""
Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
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


"""
Our code
"""


class MultiModalTransformer(nn.Module):
    """
    Embeds the shipyard information, game information and feature maps
    into a vector respresentation that is then fed into an transformere ncoder.
    Finally, a classifier head is applied for each shipyard

    Model input:
    - feature maps: (batch_size, channel, width, height)
    - shipyards: (batch_size, max_num_shipyards, num_shipyards_scalars)
    - game_stats: (batch_size, num_game_stats)

    Model output:
    - Predictions for shipyards: (batch_size, max_num_shipyards, logits)


    """

    def __init__(self, num_actions,
                 num_channels,
                 num_shipyards_scalars,
                 num_game_stats):
        super().__init__()
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
        self.transformer = Transformer(64, 8, 2, 64, 64, 0)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, num_actions)
        )
        # TODO add softmax?

    def forward(self, maps, shipyards, game_stats):
        x = self.embedding(maps, shipyards, game_stats)
        x = self.transformer(x)

        _, num_shipyards, _ = shipyards.shape
        shipyard_hidden_states = x[:,:num_shipyards]

        return self.classifier_head(shipyard_hidden_states)


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
            SmallMLP(patch_dim, embedding_dim)
        )

        self.shipyards_to_embeddings = SmallMLP(num_shipyard_scalars, embedding_dim)
        self.game_stats_to_embedding = SmallMLP(num_game_stats, embedding_dim)

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


class SmallMLP(nn.Module):
    """
    Different instances of this class are used to create the transformer inputs
    from patches, shipyard information and game information
    """

    def __init__(self, input_dim, output_dim, intermediate_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        self.fc2 = nn.Linear(intermediate_dim, intermediate_dim)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.0)
        self.fc3 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size = 2
game_stats_num = 5
shipyard_stats = 5
max_shipyards = 10

feature_maps = torch.rand((batch_size, 5, 21, 21))
shipyards = torch.randn((batch_size, max_shipyards, shipyard_stats))
game_stats = torch.randn((batch_size, game_stats_num))

transformer = MultiModalTransformer(6,
                 5,
                 5,
                 5)
print(transformer(feature_maps, shipyards, game_stats).shape)

