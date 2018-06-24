import numpy as np
import math
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_training = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
  p[range(num_training), y] -= 1
  grads = X.T.dot(p) #shape is 3073 by 10
#  p = exp_scores / np.sum(exp_scores)
  # scores is [N, C]
  for i in range(num_training):
    loss -= scores[i][y[i]]
    tempLoss = 0
    for j in range(scores.shape[1]):
      tempLoss += pow(math.e, scores[i][j]) 
#      grad = p[i, j]      
#      grad = p[:, i]
      grad = grads[:, j]
#      grad = -grads[:, i] * grads[:, j]
#        grad -= 1
#        grad = grads[:, i] * (1 - grads[:, j])

#      dW[:, j] = np.ones(X[i].shape) * grad.shape[0]
      # p is 500 by 10
      # X is 500 by 3073
      # dW[:, j] must be of length 3073
      # scores is 500 by 10 
      # dW is 3073 by 10
#      dW[:, j] = X[i] * grad 
      dW[:, j] = grad
    loss += math.log(tempLoss)

  loss /= num_training
  loss += reg * 0.5 * np.sum(W * W)
  dW /= num_training
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  num_training = X.shape[0]
  p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  p[range(num_training), y] -= 1
  grads = X.T.dot(p)
  dW = grads / num_training + reg * W
  subtractScore = scores[range(num_training), y]
#  subtractScore = subtractScore.reshape(subtractScore.shape[0], 1)
  firstLoss = np.sum(-1 * subtractScore) 
  secondLossMatrix = np.sum(np.exp(scores), axis=1) 
  secondLossMatrix = np.log(secondLossMatrix)  
  loss = firstLoss + np.sum(secondLossMatrix)
  loss /= num_training
  loss += reg * 0.5 * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

