import torch
import torchvision
import numpy as np
import pickle
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vgg

def load_data(file_list):
    data = []
    labels = []
    mask = []
    sample_frac = 0.5
    for f in file_list:
        filename = os.path.join('data', 'cifar-10-batches-py', f)
        fo = open(filename, 'rb')
        data_batch = pickle.load(fo)
        img_array = data_batch['data']
        img_array = img_array.reshape(-1,3,32,32)
        data.append(img_array)
        labels += data_batch['labels']

    data = torch.Tensor(np.concatenate(data))
    data.type(torch.FloatTensor)
    return zip(data, labels)

train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_files = ['test_batch']
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()

batch_size = [256, 2000]
lrs = [1e-4, 1e-3]
momentums = [0.9, 0.99]
n_epochs = [90,300]
interval = [50,5]
reps = 5

for i in range(2):
    trainset = load_data(train_files)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size[i], shuffle=True, num_workers=2)

    testset = load_data(test_files)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size[i], shuffle=True, num_workers=2)


    for r in range(reps):
        net = vgg.vgg16()
        net = net.cuda()
        criterion = criterion.cuda()

        optimizer = optim.SGD(net.parameters(), lr=lrs[i], momentum=momentums[i])
        if i==0:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)

        for epoch in range(n_epochs[i]):
            running_loss = 0.0
            for j, data in enumerate(trainloader,0):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                if (j+1)%interval[i]==0:
                    print '[%d, %5d] loss: %.5f' % (epoch+1,j+1, running_loss/interval[i])
                    running_loss = 0.0
            if i==0:
                scheduler.step()

        print 'Finished Training Rep %d with batch size %d' % (r+1, batch_size[i])

        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print 'GroundTruth: ' + ' '.join('%5s' % classes[labels[j]] for j in range(4))

        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data,1)
        print 'Predicted: ' + ' '.join('%5s' % classes[predicted[j]] for j in range(4))

        correct = 0
        total = 0
        for data in trainloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        print 'Accuracy of the network on the 50000 train images: %f %%' % (100. * correct / total)

        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        print 'Accuracy of the network on the 10000 test images: %f %%' % (100. * correct / total)

        weights, biases = [],[]
        for l in net.features:
            if hasattr(l, 'weight'):
                weights.append(l.weight.data.cpu().numpy())
                biases.append(l.bias.data.cpu().numpy())

        for l in net.classifier:
            if hasattr(l, 'weight'):
                weights.append(l.weight.data.cpu().numpy())
                biases.append(l.bias.data.cpu().numpy())

        net_weights = {'weights':weights, 'biases':biases}
        with open('vgg_weights_%d_rep_%d.pkl'%(batch_size[i], r+1),'wb') as f:
            pickle.dump(net_weights, f)
