from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("0_normalization", nn.BatchNorm2d(channels)),
                    ("1_activation", nn.ReLU(inplace=True)),
                    (
                        "2_convolution",
                        nn.Conv2d(
                            channels, channels, (3, 3), stride=1, padding=1, bias=False
                        ),
                    ),
                    ("3_normalization", nn.BatchNorm2d(channels)),
                    ("4_activation", nn.ReLU(inplace=True)),
                    ("5_dropout", nn.Dropout(dropout, inplace=True)),
                    (
                        "6_convolution",
                        nn.Conv2d(
                            channels, channels, (3, 3), stride=1, padding=1, bias=False
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, dropout: float
    ):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(
            OrderedDict(
                [
                    ("0_normalization", nn.BatchNorm2d(in_channels)),
                    ("1_activation", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "0_convolution",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            (3, 3),
                            stride=stride,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("1_normalization", nn.BatchNorm2d(out_channels)),
                    ("2_activation", nn.ReLU(inplace=True)),
                    ("3_dropout", nn.Dropout(dropout, inplace=True)),
                    (
                        "4_convolution",
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            (3, 3),
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                ]
            )
        )
        self.downsample = nn.Conv2d(
            in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False
        )

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        depth: int,
        dropout: float,
    ):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *(BasicUnit(out_channels, dropout) for _ in range(depth))
        )

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        width_factor: int,
        dropout: float,
        in_channels: int,
        labels: int,
    ):
        super(WideResNet, self).__init__()

        self.filters = [
            16,
            1 * 16 * width_factor,
            2 * 16 * width_factor,
            4 * 16 * width_factor,
        ]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(
            OrderedDict(
                [
                    (
                        "0_convolution",
                        nn.Conv2d(
                            in_channels,
                            self.filters[0],
                            (3, 3),
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        "1_block",
                        Block(
                            self.filters[0],
                            self.filters[1],
                            1,
                            self.block_depth,
                            dropout,
                        ),
                    ),
                    (
                        "2_block",
                        Block(
                            self.filters[1],
                            self.filters[2],
                            2,
                            self.block_depth,
                            dropout,
                        ),
                    ),
                    (
                        "3_block",
                        Block(
                            self.filters[2],
                            self.filters[3],
                            2,
                            self.block_depth,
                            dropout,
                        ),
                    ),
                    ("4_normalization", nn.BatchNorm2d(self.filters[3])),
                    ("5_activation", nn.ReLU(inplace=True)),
                    # TODO: "I think you need to print out the output sizes on each layer of the resnet, so you will have better idea how to adjust these kernel sizes given the input images"
                    # TODO: Make kernel_size an argument
                    ("6_pooling", nn.AvgPool2d(kernel_size=3)), # default is 8, changed to deal with small sizes
                    ("7_flattening", nn.Flatten()),
                    (
                        "8_classification",
                        nn.Linear(in_features=self.filters[3], out_features=labels),
                    ),
                ]
            )
        )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)
