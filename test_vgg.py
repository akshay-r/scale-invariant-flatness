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


batch_size = [256, 2000]
reps = 5

testloader = torch.utils.data.DataLoader(zip(test_x, test_y), batch_size=4, shuffle=True, num_workers=2)

for bs in batch_size:
    for r in range(reps):
        net = vgg.vgg16()
        net.cuda()
        weights_file = 'pytorch_networks/vgg_weights_%d_rep_%d.pkl'%(bs, r+1)
        print weights_file
        with open(weights_file, 'rb') as f:
            d = pickle.load(f)
            weights = d['weights']
            biases = d['biases']

        l_counter = 0
        for l in net.features:
            if hasattr(l, 'weight'):
                l.weight.data.copy_(torch.Tensor(weights[l_counter]))
                l.bias.data.copy_(torch.Tensor(biases[l_counter]))
                l_counter +=1
        for l in net.classifier:
            if hasattr(l, 'weight'):
                l.weight.data.copy_(torch.Tensor(weights[l_counter]))
                l.bias.data.copy_(torch.Tensor(biases[l_counter]))
                l_counter +=1

        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        print 'Accuracy of the network on the 10000 test images: %f %%' % (100. * float(correct) / float(total))
