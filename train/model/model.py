import torch
import torch.nn as nn
import train.model.basicblock as B


class ConvBNReLU(nn.Module):
    """
    PyTorch Module grouping together a 2D CONV, BatchNorm and ReLU layers.
    This will simplify the definition of the DnCNN network.
    """

    def __init__(
        self, in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3
    ):
        """
        Constructor
        Args:
            - in_channels: number of input channels from precedding layer
            - out_channels: number of output channels
            - kernel_size: size of conv. kernel
            - stride: stride of convolutions
            - padding: number of zero padding
        Return: initialized module
        """
        super(__class__, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward-passes input x
        """
        out = self.conv(x)
        # out = self.bn(out)
        out = self.relu(out)

        return out


class DnCNN(nn.Module):
    """
    PyTorch module for the DnCNN network.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_layers=17,
        features=64,
        kernel_size=3,
        residual=True,
    ):
        """
        Constructor for a DnCNN network.
        Args:
            - in_channels: input image channels (default 1)
            - out_channels: output image channels (default 1)
            - num_layers: number of layers (default 17)
            - num_features: number of hidden features (default 64)
            - kernel_size: size of conv. kernel (default 3)
            - residual: use residual learning (default True)
        Return: network with randomly initialized weights
        """
        super(__class__, self).__init__()

        self.residual = residual

        # a list for the layers
        self.layers = []

        # first layer
        self.layers.append(
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )
        # intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(
                ConvBNReLU(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
            )
        # last layer
        self.layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )
        # chain the layers
        self.dncnn = nn.Sequential(*self.layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward operation of the network on input x."""
        out = self.dncnn(x)

        if self.residual:  # residual learning
            out = x - out

        return self.relu(out)


class UNet(nn.Module):
    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=2,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super(UNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], mode="C" + act_mode[-1])

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = B.downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = B.downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError(
                "downsample mode [{:s}] is not found".format(downsample_mode)
            )

        self.m_down1 = B.sequential(
            *[B.conv(nc[0], nc[0], mode="C" + act_mode) for _ in range(nb)],
            downsample_block(nc[0], nc[1], mode="2" + act_mode)
        )
        self.m_down2 = B.sequential(
            *[B.conv(nc[1], nc[1], mode="C" + act_mode) for _ in range(nb)],
            downsample_block(nc[1], nc[2], mode="2" + act_mode)
        )
        self.m_down3 = B.sequential(
            *[B.conv(nc[2], nc[2], mode="C" + act_mode) for _ in range(nb)],
            downsample_block(nc[2], nc[3], mode="2" + act_mode)
        )

        self.m_body = B.sequential(
            *[B.conv(nc[3], nc[3], mode="C" + act_mode) for _ in range(nb + 1)]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = B.upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )

        self.m_up3 = B.sequential(
            upsample_block(nc[3], nc[2], mode="2" + act_mode),
            *[B.conv(nc[2], nc[2], mode="C" + act_mode) for _ in range(nb)]
        )
        self.m_up2 = B.sequential(
            upsample_block(nc[2], nc[1], mode="2" + act_mode),
            *[B.conv(nc[1], nc[1], mode="C" + act_mode) for _ in range(nb)]
        )
        self.m_up1 = B.sequential(
            upsample_block(nc[1], nc[0], mode="2" + act_mode),
            *[B.conv(nc[0], nc[0], mode="C" + act_mode) for _ in range(nb)]
        )

        self.m_tail = B.conv(nc[0], out_nc, bias=True, mode="C")
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x0, device=None):

        x0 = x0.type(torch.float)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1) + x0
        return self.relu(x)
