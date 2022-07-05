import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of ResNets based on the exercise solution. Adjusted input channel size, 
input width and height and the last layer to output a configurable size hidden representation.

(In the torchvision implementation you can't easily adjust these parameters)
"""


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()

        # The standard conv bn relu conv bn block
        self.conv1 = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(
            channels_out, channels_out, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)

        # The skip connection across the convs.
        # If the stride of this block is > 1, or the in and out channel
        # counts don't match, we need an additional 1x1 conv on the
        # skip connection.
        if stride > 1 or channels_in != channels_out:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    channels_in, channels_out, kernel_size=1, stride=stride,
                    bias=False),
                nn.BatchNorm2d(channels_out)
            )
        else:
            self.skip = nn.Sequential()

        # moved init of conv layers directly into init function
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        output = residual + self.skip(x)
        output = F.relu(output)
        return output


class ResNet18(nn.Module):
    # 1 conv + 4 * 2 blocks (=2 convs) + 1 final 'conv' (=fc) -> ResNet 18

    def __init__(self, input_channels=1, output_dim=120):
        super().__init__()
        # Internal varialbe that keeps track of the channels going to the next
        # section of resblocks.
        self.channels_in = 64

        # The first input convolution, going from 3 channels to 64.
        self.conv1 = nn.Conv2d(
            input_channels, self.channels_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels_in)

        # Four "sections", each with 2 resblocks and increasing channel counts.
        self.section1 = self._add_section(64, 2, stride=1)
        self.section2 = self._add_section(128, 2, stride=2)
        self.section3 = self._add_section(256, 2, stride=2)
        self.section4 = self._add_section(512, 2, stride=2)

        # The final linear layer to get the logits. This could also
        # be realized with a 1x1 conv.
        self.linear = nn.Linear(512, output_dim)

    # Utility function to add a section with a fixed amount of res blocks
    def _add_section(self, channels_out, resblock_count, stride):
        resblocks = []
        for b in range(resblock_count):
            stride_block = stride if b == 0 else 1
            resblocks.append(
                ResBlock(self.channels_in, channels_out, stride=stride_block))
            self.channels_in = channels_out
        return nn.Sequential(*resblocks)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x.shape: (b, 64, 21, 21)
        x = self.section1(x)
        # x.shape: (b, 64, 21, 21)
        x = self.section2(x)
        # x.shape: (b, 128, 11, 11)
        x = self.section3(x)
        # x.shape: (b, 256, 6, 6)
        x = self.section4(x)
        # x.shape: (b, 512, 3, 3)
        x = torch.mean(x, dim=(2, 3))
        x = self.linear(x)
        return x
