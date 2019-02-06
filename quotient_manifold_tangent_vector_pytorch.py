import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class QuotientManifoldTangentVector(object):
    """
    Container class for neural network parameter vectors represented
    on the Quotient Manifold
    """
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_components = len(layer_sizes)
        self.vec = [np.zeros(self.layer_sizes[i]) for i in range(self.n_components)]

    def set_vector(self, values, overwrite=False):
        if len(values) != self.n_components:
            if overwrite:
                self.vec = values
                self.n_components = len(values)
            else:
                raise Exception('This vector has been initialized with %d components \
                                 and %d components have been provided' % (self.n_components, len(values)))
        else:
            self.vec = values

    def get_vector(self):
        return self.vec

    def dot(self, b, weights):
        if not isinstance(b, QuotientManifoldTangentVector):
            raise Exception('Cannot find dot product with non QuotientManifoldTangentVector quantity')
        if not isinstance(weights, QuotientManifoldTangentVector):
            raise Exception('Weight vector is not a QuotientManifoldTangentVector')
        if b.n_components != self.n_components:
            raise Exception('Both QuotientManifoldTangentVectors need to have same number of components')
        if weights.n_components != self.n_components:
            raise Exception('Weight QuotientManifoldTangentVector needs to have same number of components')

        dot_prod = np.sum([np.dot(self.vec[i].ravel(),b.vec[i].ravel())/(np.linalg.norm(weights.vec[i].ravel())**2) for i in range(self.n_components)])
        return dot_prod

    def norm(self, weights):
        return np.sqrt(self.dot(self, weights))

    def normalize(self, weights):
        N = self.norm(weights)
        normed_vec = [z/N for z in self.vec]
        self.set_vector(normed_vec)

    def riemannian_hess_vec_prod(self, func, weight_tensors):
        if len(weight_tensors) != self.n_components:
            raise Exception('Mismatch between number of tangent vector components and weight tensors provided')

        grads = torch.autograd.grad(func, weight_tensors, create_graph=True)
        if torch.cuda.is_available():
            g_v_prod = sum([torch.dot(grads[i].view(-1), torch.Tensor(self.vec[i]).cuda().view(-1)) for i in range(self.n_components)])
        else:
            g_v_prod = sum([torch.dot(grads[i].view(-1), torch.Tensor(self.vec[i]).view(-1)) for i in range(self.n_components)])
        hess_vec_prod = torch.autograd.grad(g_v_prod, weight_tensors)
        norms = [torch.norm(var)**2 for var in weight_tensors]
        r_hess_vec_prod = [np.copy((hess_vec_prod[i]*norms[i]).data.cpu().numpy()) for i in range(len(norms))]
        return r_hess_vec_prod

def riemannian_hess_quadratic_form(tgt_vec, net, criterion, weights, data, labels):
    V = QuotientManifoldTangentVector(weights.layer_sizes)
    V.set_vector(tgt_vec.vec)

    if torch.cuda.is_available():
        inputs, targets = Variable(torch.Tensor(data).cuda()), Variable(torch.Tensor(labels).type(torch.LongTensor).cuda())
        net = net.cuda()
        criterion = criterion.cuda()
    else:
        inputs, targets = Variable(torch.Tensor(data)), Variable(torch.Tensor(labels).type(torch.LongTensor))
    loss = criterion(net(inputs), targets)

    hv_prod = V.riemannian_hess_vec_prod(loss, net.get_weight_tensors())
    HV = QuotientManifoldTangentVector(weights.layer_sizes)
    HV.set_vector(hv_prod)
    return V.dot(HV, weights)

def riemannian_power_method(v_init, max_iter, net, criterion, weights, data, labels, tol=1e-8):
    V_T = QuotientManifoldTangentVector(weights.layer_sizes)
    V_T.set_vector(v_init)
    V_Tp1 = QuotientManifoldTangentVector(weights.layer_sizes)
    errs = np.zeros(max_iter)
    for i in range(max_iter):
        if torch.cuda.is_available():
            inputs, targets = Variable(torch.Tensor(data).cuda()), Variable(torch.Tensor(labels).type(torch.LongTensor).cuda())
            net = net.cuda()
            criterion = criterion.cuda()
        else:
            inputs, targets = Variable(torch.Tensor(data)), Variable(torch.Tensor(labels).type(torch.LongTensor))
        loss = criterion(net(inputs), targets)
        net.zero_grad()
        v_tp1 = V_T.riemannian_hess_vec_prod(loss, net.get_weight_tensors())
        V_Tp1.set_vector(v_tp1)
        V_Tp1.normalize(weights)

        err = np.sqrt(sum([np.linalg.norm(a.ravel() - b.ravel())**2 for a,b in zip(V_Tp1.vec,V_T.vec)]))/np.sqrt(sum([np.linalg.norm(z.ravel())**2 for z in V_T.vec]))
        V_T.set_vector(V_Tp1.vec)
        errs[i] = err
        if err < tol:
            break
    return V_T, errs
