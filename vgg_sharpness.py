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
from quotient_manifold_tangent_vector_pytorch import QuotientManifoldTangentVector,\
                                                     riemannian_power_method,\
                                                     riemannian_hess_quadratic_form

from load_cifar_data import load_cifar_as_array

train_x, train_y, val_x, val_y, test_x, test_y = load_cifar_as_array('data/cifar-10-batches-py/')

batch_size = [256, 2000]
reps = 5

sharpness = []
losses = []
norms = []
for bs in batch_size:
    for r in range(reps):
        net = vgg.vgg16()
        criterion = nn.CrossEntropyLoss()

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

        layer_sizes = [w.shape for w in weights] + [b.shape for b in biases]
        v_init = [np.random.normal(size=layer_sizes[i]) for i in range(len(layer_sizes))]

        W_orig = QuotientManifoldTangentVector(layer_sizes)
        W_orig.set_vector(weights+biases)

        n_samples = 10000

        t1 = time.time()
        v_res,errs = riemannian_power_method(v_init, 1000, net, criterion, W_orig, train_x[:n_samples], train_y[:n_samples], tol=1e-6)
        sp_norm = riemannian_hess_quadratic_form(v_res, net, criterion, W_orig, train_x[:n_samples], train_y[:n_samples])
        secs1 = time.time() - t1
        print 'Measuring sharpness took %.4f seconds' % (secs1)

        if torch.cuda.is_available():
            inputs, labels = Variable(torch.Tensor(train_x[:n_samples]).cuda()), Variable(torch.Tensor(train_y[:n_samples]).type(torch.LongTensor).cuda())
            net = net.cuda()
            criterion = criterion.cuda()
        else:
            inputs, labels = Variable(torch.Tensor(train_x[:n_samples])), Variable(torch.Tensor(train_y[:n_samples]).type(torch.LongTensor))

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sharpness.append(sp_norm/(2+2*loss.data.cpu().numpy()))

print sharpness
