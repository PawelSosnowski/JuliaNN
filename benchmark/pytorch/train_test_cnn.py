from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torchvision import datasets, transforms

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
        self.fc1 = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x
    
def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        n = module.weight.size(1)
        stdv = 1. / math.sqrt(n)
        module.weight.data.uniform_(-stdv, stdv)
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.Conv2d):
        n = module.in_channels
        stdv = 1. / math.sqrt(n)
        module.weight.data.uniform_(-stdv, stdv)

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, scheduler: LRScheduler, epoch: int, device: torch.device):
    model.train()
    
    for batch_index, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target, reduction='mean')

        loss.backward()

        if batch_index % 10 == 0:
            print(f'Train Epoch: {epoch} ({100. * batch_index / len(train_loader):.0f}%) Loss: {loss.item():.6f}')
        
        optimizer.step()
    
    scheduler.step()

def evaluate(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            prediction = output.argmax(dim=1, keepdim=True)
            test_accuracy += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)

    print(f'Test loss: {test_loss:.3f}, test accuracy: {test_accuracy:.3f}')

def load_and_transform_mnist()-> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    train_dataset = datasets.MNIST('benchmark/data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('benchmark/data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    return train_loader, test_loader


torch.manual_seed(0)

if __name__ == '__main__':
    torch.set_num_threads(1)
    device = torch.device('cpu')

    train_data, test_data = load_and_transform_mnist()
    model = CNNet().to(device)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, 11):
        train(model, train_data, optimizer, lr_scheduler, epoch, device)
        evaluate(model, test_data, device)