import torch
import torch.nn as nn

class Branch(nn.Module):
  def __init__(self, in_channels):
    self.features = nn.Sequential(
        self._make_layer(2, in_channels, 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        self._make_layer(2, 64, 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        self._make_layer(3, 128, 256),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        self._make_layer(3, 256, 512),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        self._make_layer(3, 512, 512, dilation=2),
        conv1x1(512, 256),
        # nn.ReLU(inplace=True)
    )

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)
