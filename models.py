import torch.nn as nn
import torch.nn.functional as F

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)     # input channel = 3(RGB), output channel = 6, kernel size = 5x5
#         self.pool = nn.MaxPool2d(2,2)       # filter size 2x2
#         self.conv2 = nn.Conv2d(6, 16, 5)    # output channel = 16
#         self.fc1 = nn.Linear(16*5*5, 120)   # in: 16xkernelxkernel, out features = 120    --> flattening
#         self.fc2 = nn.Linear(120, 84)       # in: 120, out features = 84
#         self.fc3 = nn.Linear(84, 10)        # in: 84, out features = 10 (10 classes in CIFAR10)
#         # values 120 and 84 are arbitrary values choses from experimentation on CIFAR10 datasets.
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))    # conv(x) --> relu --> maxpooling   (output channel = 6)
#         x = self.pool(F.relu(self.conv2(x)))    # conv(x) --> relu --> maxpooling   (output channel = 16)
#         x = x.view(-1, 16*5*5)                  # reshaping 3D tensor of size 16*5*5 to 1D tensor
#         x = F.relu(self.fc1(x))                 # relu activation applied to introduce non-linearity.
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )