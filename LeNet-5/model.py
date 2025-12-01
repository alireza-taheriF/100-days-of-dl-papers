import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.s2(x)
        x = self.act(self.c3(x))
        x = self.s4(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.c5(x))
        x = self.act(self.f6(x))
        x = self.output(x)
        return x