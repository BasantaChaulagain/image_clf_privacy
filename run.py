import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from models import convnet
import timm


# constants
MAX_GRAD_NORM = 1.0    # The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step
# NOISE_MULTIPLIR = 1.3  # The ratio of (sd of noise added to the gradients) to (the sensitivity of gradients)
EPSILON = 5.0
DELTA = 1e-5        
EPOCHS = 200
LR = 1e-3               # Learning rate of the algorithm
BATCH_SIZE = 128

# max_grad_norm, epsilon, lr
# constants = [[2.5, 3.0, 5e-4], [5.0, 3.0, 5e-4]]
# M = 0.9                 # momentum
# WD = 0.1                # weight decay

# These values, specific to the CIFAR10 dataset, are assumed to be known.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])
    train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform)
    test_ds = CIFAR10(root='data/', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, pin_memory=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)
    
    return trainloader, testloader

def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, trainloader, optimizer, epoch, privacy_engine, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses, top1_acc = [], []
    
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
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()
            acc = accuracy(preds, labels)
            losses.append(loss.detach().cpu().numpy())
            top1_acc.append(acc)
    
    top1_avg = np.mean(top1_acc)
    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)


def main():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    trainloader, testloader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(1):
        print("run:", i)
        
        # array of all the models used for training
        # models = []
        # models.append(convnet(10))
        # models.append(torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True))
        # models.append(torchvision.models.densenet161(pretrained=True))
        # models.append(torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT))
        # models.append(EfficientNet.from_pretrained('efficientnet-b0', num_classes=10))
        # models.append(ViT(image_size=32, patch_size=4, num_classes=10, dim=64, depth=6, heads=8, mlp_dim=128))
    
        # for const in constants:
        # for model in models:
            # MAX_GRAD_NORM = const[0]
            # EPSILON = const[1]
            # LR = const[2]
        model = torchvision.models.densenet161(pretrained=True)
        # if model.__class__.__name__ == "Sequential":
        #     EPOCHS = 10
            
        print("model: ", model.__class__.__name__)
        
        model.to(device)
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            ModuleValidator.validate(model, strict=False)
        # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=M, weight_decay=WD)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
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
            
        print("\n==========\n")

if __name__ == "__main__":
    main()