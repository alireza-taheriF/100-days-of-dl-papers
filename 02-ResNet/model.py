import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(identity)

        out += shortcut

        out = self.relu(out)

        return out


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    block = BasicBlock(in_channels=64, out_channels=64, stride=1)
    y = block(x)
    print(f"Test 1 (Same Dim): Input {x.shape} -> Output {y.shape}")
    assert x.shape == y.shape, "Test 1 Failed!"

    block_down = BasicBlock(in_channels=64, out_channels=128, stride=2)
    y_down = block_down(x)
    print(f"Test 2 (Downsample): Input {x.shape} -> Output {y_down.shape}")
    assert y_down.shape == (2, 128, 16, 16), "Test 2 Failed!"

    print("✅ BasicBlock implemented correctly!")