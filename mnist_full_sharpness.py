import numpy as np
import tensorflow as tf
from copy import copy

from tensorflow_networks.mlp_architectures import MLP_model
from quotient_manifold_tangent_vector import QuotientManifoldTangentVector,\
                                             riemannian_power_method,\
                                             riemannian_hess_quadratic_form
from load_data import load_data_as_ndarray

test_x, test_labels, val_x, val_labels, train_x, train_labels = load_data_as_ndarray('data/mnist.pkl.gz')


def labels_to_onehot(labels, digits):
    idx = [l for l in xrange(len(labels)) if labels[l] in digits]
    ids = np.eye(10)
    ids = ids[:,digits]
    out_data = ids[labels[idx],:]
    return out_data

digits = range(10)
test_y = labels_to_onehot(test_labels, digits)
val_y = labels_to_onehot(val_labels, digits)
train_y = labels_to_onehot(train_labels, digits)

sess = tf.Session()
input_dim = train_x.shape[1]
n_classes = len(digits)
hidden_layers = [512]*5
lr = 0.001

test_acc = {}
sharpness = {}
reps = 5
n_epochs = 200
batch_size = [256, 5000]
eps = 1e-3
for b in batch_size:
    test_acc[b] = []
    sharpness[b] = []
    for r in range(reps):
        print('batch size: %d, rep:%d' % (b, r+1))
        model = MLP_model(session=sess, input_dim=input_dim, n_classes=n_classes, include_bias=False,
                          hidden_layers=copy(hidden_layers), output_nonlinearity='softmax')
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        model.compile(optimizer=opt)
        model.fit_evaluate(train_x, train_y, val_x, val_y, n_epochs, b, verbose=1)
        test_pred = model.predict(test_x, b)
        test_acc[b].append(model.accuracy(test_y, test_pred))

        W = model.get_trainable_weights()
        layer_sizes = [w.shape for w in W]
        v_init = [np.random.normal(scale=1./np.sqrt(layer_sizes[i][1]), size=layer_sizes[i]) for i in range(len(layer_sizes))]

        W_orig = QuotientManifoldTangentVector(layer_sizes, with_tensors=False)
        W_orig.set_vector(W)
        model.set_weights(W_orig.vec)
        v_res,errs = riemannian_power_method(v_init, 1000, model, W_orig, train_x, train_y)
        sp_norm = riemannian_hess_quadratic_form(v_res, model, W_orig, train_x, train_y)
        sharpness[b].append((sp_norm * eps**2)/(2+2*model.get_loss_value(train_x, train_y)))

import pickle
with open('mnist_full_test_acc.pkl','wb') as f:
    pickle.dump(test_acc, f)

with open('mnist_full_sharpness.pkl','wb') as f:
    pickle.dump(sharpness, f)
