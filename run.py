import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from opacus import PrivacyEngine
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

class Net(nn.Module):
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


def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct/total
    print("Accuracy on the test images: ", accuracy)


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform)
    test_ds = CIFAR10(root='data/', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(train_ds, batch_size=8, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # net.parameters() returns weights and biases of nn. (out channel, in channel, kernel, kernel) eg: (6, 3, 5, 5) for conv1(x)
    
    epsilon = 1.0
    delta = 1e-5

    privacy_engine = PrivacyEngine()
    net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module = net,
        optimizer = optimizer,
        data_loader = trainloader,
        noise_multiplier = 1.3,             # ratio of (sd of noise added to the gradients) to (the sensitivity of gradients)
        max_grad_norm = 1.5,                # maximum L2 norm of gradients before clipping.
    )
    
    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
            
            if i%2000 == 1999:
                eps, delt = privacy_engine.get_privacy_spent(delta)
                print('eps: %.3f delt: %.3f' % (eps, delt))
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
    
    evaluate(net, testloader)


if __name__ == "__main__":
    main()