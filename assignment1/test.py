import numpy as np
X = np.ones((500, 3000))
X_train = np.ones((5000, 3000))
num_test = X.shape[0]
num_train = X_train.shape[0]
dists = np.zeros((num_test, num_train))

# broadcastX = X[..., None, :]
# broadcastX_train = X_train[None, :, :]

# dists = (broadcastX ** 2) + (broadcastX_train ** 2) - (2 * broadcastX * broadcastX_train)

dists = ((X ** 2).sum(axis=1, keepdims=True) + (X_train ** 2).sum(axis=1)) - np.dot(X, X_train.T)
print(dists.shape)
