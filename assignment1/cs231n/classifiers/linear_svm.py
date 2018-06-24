import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    arrayWrong = ((scores - correct_class_score + 1) > 0) * 1 
    arrayWrong[y[i]] = 0
    numberWrong = sum(arrayWrong)

    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += -X[i] * numberWrong 
        continue
      
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:, j] += X[i]        
        loss += margin
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train 
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW += reg * 2 * W 
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dWmid = np.zeros(W.shape)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)  
  # scores is N by C
  mask = range(scores.shape[0])
  correct_class_scores = scores[mask, y[mask]] 
  correct_class_scores = correct_class_scores.reshape(correct_class_scores.shape[0], 1)

  allWrong = ((scores - correct_class_scores + 1) > 0) * 1
  allWrong[mask, y[mask]] = 0
  allMargins = allWrong * (scores - correct_class_scores + 1)   
  loss = np.sum(allMargins)
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)


  testWrong = np.zeros(allWrong.shape)
  testWrong[mask, y[mask]] = 1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  sumX = np.sum(X, axis=0)
  sumX = sumX.reshape(sumX.shape[0], 1)
#  dW += sumX  
  dW += X.T.dot(allWrong)  
  # all wrong has shape of scores
  numberWrong = np.sum(allWrong, axis=1) 
  numberWrong = numberWrong.reshape(numberWrong.shape[0], 1)

  # number wrong is is [n, 1]
  # contains number wrong for each one to use for the correct class
  # correct class uses y
#  for i in mask:
#    dW[:, y[i]] += -X[i] * numberWrong[i]
  temp = -X * numberWrong  
  dW += temp.T.dot(testWrong)  


  dW /= X.shape[0]
  dW += reg * 2 * W 
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
