**A Scale Invariant Measure of Flatness for Deep Network Minima**

This repository contains the code used to run the experiments in the paper "A Scale Invariant Measure of Flatness for Deep Network Minima"

The key algorithm in the paper is implemented in Pytorch as well as Tensorflow in the files `quotient_manifold_tangent_vector_pytorch.py` and  `quotient_manifold_tangent_vector.py` respectively.
The relevant functions are `riemannian_power_method` and `riemannian_hess_quadratic_form`

The jupyter notebook included here serves as a guide to use our algorithm. The simulations in section 3 were run using a version of this notebook

The mnist experiments were run in Tensorflow using the scripts `mnist_full_sharpness.py` and `mnist_conv_sharpness.py`

The cifar10 experiments were run in pytorch. The networks were trained using the scripts in `alexnet_training_all.py` located in the folder `pytorch_networks`
The test accuracies and sharpness measurements were made using the scripts `test_alexnet.py` and `alexnet_sharpness.py`.
Similar experiments were run using the vgg files. The vgg network was implemented in `pytorch_networks/vgg.py`.
