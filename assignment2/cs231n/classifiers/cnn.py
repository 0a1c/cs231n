from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.weight_scale = weight_scale
        self.hidden_dim = hidden_dim

        C, H, W = input_dim
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        W1 = np.random.normal(loc=0, scale=weight_scale, size=(num_filters, C, filter_size, filter_size))
        b1 = np.zeros(num_filters)
        # result from first will be of shape N, F, H', W'
        # convert to N by ___
        W2 = np.random.normal(loc=0, scale=weight_scale, size=(1, hidden_dim))
#        W2 = None
        b2 = np.zeros(hidden_dim)
        
        W3 = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, num_classes))
        b3 = np.zeros(num_classes)


        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['W3'] = W3
        self.params['b1'] = b1
        self.params['b2'] = b2
        self.params['b3'] = b3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        N = X.shape[0]
        reg = self.reg
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        #conv - relu - 2x2 max pool - affine - relu - affine - softmax
        first_out, first_cache = conv_relu_forward(X, W1, b1, conv_param)
        second_out, second_cache = max_pool_forward_fast(first_out, pool_param)
        W2_input_shape = second_out.reshape(second_out.shape[0], -1).shape[1]
        if W2.shape[0] == 1:
            W2 = np.random.normal(loc=0, scale=self.weight_scale, size=(W2_input_shape, self.hidden_dim))
            self.params['W2'] = W2
        third_out, third_cache = affine_relu_forward(second_out, W2, b2)
        scores, final_cache = affine_forward(third_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

        dx, dw3, db3 = affine_backward(dx, final_cache)
        dx, dw2, db2 = affine_relu_backward(dx, third_cache)
        dx = max_pool_backward_fast(dx, second_cache)
        dx, dw1, db1 = conv_relu_backward(dx, first_cache)

        dw1 += reg * W1
        dw2 += reg * W2
        dw3 += reg * W3
        db1 += reg * b1
        db2 += reg * b2
        db3 += reg * b3

        grads['W1'] = dw1 
        grads['W2'] = dw2 
        grads['W3'] = dw3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
