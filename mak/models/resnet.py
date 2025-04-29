"""Implements the Base Resnet model."""

from typing import Tuple

import torch.nn.functional as F
from torch import Tensor, nn

from mak.models.base_model import Model


class BasicBlock(nn.Module):
    """Resnet Basic Block. (https://arxiv.org/abs/2103.16257)"""

    expansion = 1

    def __init__(
        self, in_planes: int, planes: int, activation: nn.functional, stride: int = 1
    ):
        """
        Initializes a ResNet basic block.

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            activation (nn.functional): Activation function.
            stride (int): Stride for the convolutional layers. Defaults to 1.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResNet basic block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    """Resnet Bottleneck Block."""

    expansion = 4

    def __init__(
        self, in_planes: int, planes: int, activation: nn.functional, stride: int = 1
    ):
        """
        Initializes a ResNet bottleneck block.

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            activation (nn.functional): Activation function.
            stride (int): Stride for the convolutional layers. Defaults to 1.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.activation = activation

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResNet bottleneck block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class BaseResNet(Model):
    """Resnet Base Model Class."""

    def __init__(
        self,
        block: nn.Module,
        num_blocks: list[int],
        activation: nn.functional,
        num_classes: int,
        input_shape: Tuple,
        *args,
        **kwargs
    ):
        """
        Initializes a ResNet model.

        Args:
            block (nn.Module): Block type for ResNet (e.g., BasicBlock or Bottleneck).
            num_blocks (list[int]): List specifying the number of blocks in each layer of the ResNet.
            activation (nn.functional): Activation function.
            num_classes (int): Number of output classes.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(input_shape, num_classes, *args, **kwargs)

        self.in_planes = 64
        self.activation = activation
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(
            self.input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int):
        """
        Helper function to create a ResNet layer.

        Args:
            block (nn.Module): Block type for ResNet (e.g., BasicBlock or Bottleneck).
            planes (int): Number of output channels for each block in the layer.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the convolutional layers.

        Returns:
            nn.Sequential: Sequential container for the layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for curr_stride in strides:
            layers.append(block(self.in_planes, planes, self.activation, curr_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResNet model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Resnet18(BaseResNet):
    """Resnet 18 Model Class."""

    def __init__(self, num_classes: int = 10, *args, **kwargs):
        super().__init__(
            block=BasicBlock,
            num_blocks=[2, 2, 2, 2],
            activation=F.relu,
            num_classes=num_classes,
            *args,
            **kwargs
        )

        self.__class__.__name__ = "Resnet18 Custom"


class Resnet34(BaseResNet):
    """Resnet 34 Model Class."""

    def __init__(self, num_classes: int = 10, *args, **kwargs):
        super().__init__(
            block=BasicBlock,
            num_blocks=[3, 4, 6, 3],
            activation=F.relu,
            num_classes=num_classes,
            *args,
            **kwargs
        )

        self.__class__.__name__ = "Resnet34 Custom"
