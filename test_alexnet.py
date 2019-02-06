import torch
import torchvision
import numpy as np
import time
import pickle
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pytorch_networks.vgg as vgg

from load_cifar_data import load_cifar_as_array

train_x, train_y, val_x, val_y, test_x, test_y = load_cifar_as_array('data/cifar-10-batches-py/')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(4096, 384)
        self.fc2 = nn.Linear(384,192)
        self.fc3 = nn.Linear(192,10)

    def get_weight_tensors(self):
        W = [self.conv1.weight, self.conv2.weight, self.fc1.weight, self.fc2.weight, self.fc3.weight]
        W = W + [self.conv1.bias, self.conv2.bias, self.fc1.bias, self.fc2.bias, self.fc3.bias]
        return W

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

batch_size = [256, 2000]
reps = 5

testloader = torch.utils.data.DataLoader(zip(test_x, test_y), batch_size=4, shuffle=True, num_workers=2)

for bs in batch_size:
    for r in range(reps):
        net = AlexNet()
        net.cuda()
        weights_file = 'pytorch_networks/alexnet_weights_%d_rep_%d.pkl'%(bs, r+1)
        print weights_file
        with open(weights_file, 'rb') as f:
            d = pickle.load(f)
            weights = d['weights']
            biases = d['biases']

        layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
        for i in range(len(layers)):
            net.__getattr__(layers[i]).weight.data.copy_(torch.Tensor(weights[i]))
            net.__getattr__(layers[i]).bias.data.copy_(torch.Tensor(biases[i]))

        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        print 'Accuracy of the network on the 10000 test images: %f %%' % (100. * float(correct) / float(total))
