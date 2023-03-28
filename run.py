import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

from efficientnet_pytorch import EfficientNet
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from models import ConvNet


# constants
MAX_GRAD_NORM = 1.0     # The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step
NOISE_MULTIPLIER = 1.3  # The ratio of (sd of noise added to the gradients) to (the sensitivity of gradients)
EPSILON = 20.0
DELTA = 1e-5        
EPOCHS = 10
LR = 1e-3               # Learning rate of the algorithm
BATCH_SIZE = 128
M = 0.9                 # momentum
WD = 0.1                # weight decay

# These values, specific to the CIFAR10 dataset, are assumed to be known.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])
    train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform)
    test_ds = CIFAR10(root='data/', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return trainloader, testloader

def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, trainloader, optimizer, epoch, privacy_engine, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses, top1_acc = [], []
    print(epoch)
    
    for (images, labels) in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        labels = labels.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)
        
        losses.append(loss.item())
        top1_acc.append(acc1)
        loss.backward()
        optimizer.step()
        
    epsilon = privacy_engine.get_epsilon(DELTA)
    print(
        f"\tTrain Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
        f"(ε = {epsilon:.2f}, δ = {DELTA})"
    )


def test(model, testloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses, top1_acc = [], []
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = np.argmax(outputs.numpy(), axis=1)
            labels = labels.numpy()
            acc = accuracy(preds, labels)
            losses.append(loss)
            top1_acc.append(acc)
    
    top1_avg = np.mean(top1_acc)
    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)


def main():
    trainloader, testloader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # array of all the models used for training
    models = []
    models.append(ConvNet())
    models.append(torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT))
    models.append(EfficientNet.from_pretrained('efficientnet-b0', num_classes=10))
    models.append(torchvision.models.vit_b_16())

    for model in models:
        print("model: ", model.__class__.__name__)    
        # if model.__class__.__name__ == "ConvNet" or model.__class__.__name__ =="EfficientNet":
        if model.__class__.__name__ == "ConvNet":
            continue
        
        model.to(device)
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=M, weight_decay=WD)
        
        privacy_engine = PrivacyEngine()
        model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs=EPOCHS,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
        
        for epoch in range(EPOCHS):
            train(model, trainloader, optimizer, epoch, privacy_engine, device)
        
        test(model, testloader, device)

if __name__ == "__main__":
    main()