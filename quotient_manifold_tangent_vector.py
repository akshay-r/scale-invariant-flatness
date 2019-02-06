import numpy as np
import tensorflow as tf

class QuotientManifoldTangentVector(object):
    """
    Container class for neural network parameter vectors represented
    on the Quotient Manifold
    """
    def __init__(self, layer_sizes, with_tensors=True):
        self.layer_sizes = layer_sizes
        self.n_components = len(layer_sizes)
        self.vec = [np.zeros(self.layer_sizes[i]) for i in range(self.n_components)]
        self.with_tensors = with_tensors
        if with_tensors:
            self.vec_tf = [tf.placeholder(tf.float32, shape=self.layer_sizes[i]) for i in range(self.n_components)]

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

    def riemannian_hess_vec_prod(self, func, weights_tf):
        if len(weights_tf) != self.n_components:
            raise Exception('Mismatch between number of tangent vector components and weight tensors provided')

        grads = tf.gradients(func, weights_tf)

        g_v_prod = np.sum([tf.reduce_sum(tf.multiply(self.vec_tf[i], grads[i])) for i in range(self.n_components)])
        hess_vec_prod = tf.gradients(g_v_prod, weights_tf)
        r_hess_vec_prod = [tf.multiply(hess_vec_prod[i], tf.pow(tf.norm(weights_tf[i]), tf.constant(2.))) for i in range(len(weights_tf))]
        return r_hess_vec_prod

def riemannian_hess_quadratic_form(tgt_vec, model, weights, data, labels):
    V = QuotientManifoldTangentVector(weights.layer_sizes)
    V.set_vector(tgt_vec.vec)
    hv_prod = V.riemannian_hess_vec_prod(model.loss, model.weights)
    feed_dict = {model.inputs:data, model.labels:labels}
    for j in range(V.n_components):
        feed_dict[V.vec_tf[j]] = V.vec[j]
    HV = QuotientManifoldTangentVector(weights.layer_sizes)
    HV.set_vector(model.session.run(hv_prod, feed_dict=feed_dict))
    return V.dot(HV, weights)

def riemannian_power_method(v_init, max_iter, model, weights, data, labels, tol=1e-8):
    V = QuotientManifoldTangentVector(weights.layer_sizes)
    V.set_vector(v_init)
    hv_prod = V.riemannian_hess_vec_prod(model.loss, model.weights)
    V_T = QuotientManifoldTangentVector(weights.layer_sizes)
    V_T.set_vector(v_init)
    errs = np.zeros(max_iter)
    for i in range(max_iter):
        feed_dict = {model.inputs:data, model.labels:labels}
        for j in range(V.n_components):
            feed_dict[V.vec_tf[j]] = V_T.vec[j]
        v_tp1 = model.session.run(hv_prod, feed_dict=feed_dict)
        V_Tp1.set_vector(v_tp1)
        V_Tp1.normalize(weights)

        err = np.sqrt(sum([np.linalg.norm(a.ravel() - b.ravel())**2 for a,b in zip(V_Tp1.vec,V_T.vec)]))/np.sqrt(sum([np.linalg.norm(z.ravel())**2 for z in V_T.vec]))
        V_T.set_vector(V_Tp1.vec)
        errs[i] = err
        if err < tol:
            break
    return V_T, errs
