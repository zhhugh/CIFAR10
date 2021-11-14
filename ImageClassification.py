#!/usr/bin/env python
# coding: utf-8

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import torchvision
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=False)


cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = VGG('VGG16').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 4], gamma=0.5)
criterion = nn.CrossEntropyLoss().to(device)


writer = SummaryWriter('logs')
accuracy = 0
acc = []
los = []
epochs = 10
bar = tqdm(range(1, epochs + 1), desc='epochs')
for epoch in bar:
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
        total += target.size(0)
        # train_correct incremented by one if predicted right
        train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
    print(scheduler.get_last_lr())
    scheduler.step()
    train_result,train_acc = train_loss, train_correct / total
    writer.add_scalar('train_loss', train_result, epoch)
    writer.add_scalar('trian_accuracy', train_acc, epoch)
    acc.append(train_acc)
    los.append(train_result)
    print("test:")
    model.eval()
    test_loss = 0
    test_correct = 0
    total = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            prediction = torch.max(output, 1)
            total += target.size(0)
            test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

    test_result = test_loss, test_correct / total

    print("train loss=%.5f" % train_loss)
    print("train accuracy=%.5f" % train_acc)
    print("test loss=%.5f" % test_result[0])
    print("test accuracy=%.5f" % test_result[1])
    accuracy = max(accuracy, test_result[1])
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_accuracy', accuracy, epoch)






