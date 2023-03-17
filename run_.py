import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import CIFAR10


# constants
MAX_GRAD_NORM = 1.2     # The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step
NOISE_MULTIPLIER = 1.3  # The ratio of (sd of noise added to the gradients) to (the sensitivity of gradients)
EPSILON = 50.0
DELTA = 1e-5        
EPOCHS = 20
LR = 1e-3               # Learning rate of the algorithm
BATCH_SIZE = 4

# These values, specific to the CIFAR10 dataset, are assumed to be known.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])
    train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform)
    test_ds = CIFAR10(root='data/', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return trainloader, testloader

class ConvNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # input channel = 3(RGB), output channel = 6, kernel size = 5x5
        self.pool = nn.MaxPool2d(2,2)       # filter size 2x2
        self.conv2 = nn.Conv2d(6, 16, 5)    # output channel = 16
        self.fc1 = nn.Linear(16*5*5, 120)   # in: 16xkernelxkernel, out features = 120    --> flattening
        self.fc2 = nn.Linear(120, 84)       # in: 120, out features = 84
        self.fc3 = nn.Linear(84, 10)        # in: 84, out features = 10 (10 classes in CIFAR10)
        # values 120 and 84 are arbitrary values choses from experimentation on CIFAR10 datasets.
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # conv(x) --> relu --> maxpooling   (output channel = 6)
        x = self.pool(F.relu(self.conv2(x)))    # conv(x) --> relu --> maxpooling   (output channel = 16)
        x = x.view(-1, 16*5*5)                  # reshaping 3D tensor of size 16*5*5 to 1D tensor
        x = F.relu(self.fc1(x))                 # relu activation applied to introduce non-linearity.
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def accuracy(preds, labels):
    return (preds == labels).mean()

def main():
    trainloader, testloader = load_data();
    model = ConvNet()
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    

if __name__ == "__main__":
    main()