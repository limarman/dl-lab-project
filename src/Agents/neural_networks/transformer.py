from transformers import ViTConfig, ViTModel


class VisionTransformer(ViTModel):
    """
    Simple Wrapper for the hugginface VIT implementation to
    easily retrieve the hidden state for the classification token
    (intended to use in HybridNet)
    """

    def __init__(self, num_channels, hidden_size, patch_size=3):
        config = ViTConfig(image_size=21,
                           num_channels=num_channels,
                           patch_size=patch_size,
                           hidden_size=hidden_size,
                           intermediate_size=1024) # reducing intermediate size doubles fps
        super().__init__(config)

    def forward(self, maps):
        result = super(VisionTransformer, self).forward(pixel_values=maps)
        classification_token_hidden_state = result['last_hidden_state'][:, 0, :]
        return classification_token_hidden_state
