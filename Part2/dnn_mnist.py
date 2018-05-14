from scipy import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


"""
All of the methods below are from Part1/mnist.py
"""

TRAIN_SIZE = 60000
url = 'https://github.com/amplab/datascience-sp14/blob/master/lab7/mldata/mnist-original.mat'
DL_PATH = 'datasets/mnist/mnist-original.mat'

# load the data
def load_mnist():
	return io.loadmat(DL_PATH)

# split data into data and label
def split_X_y():
	mnist = load_mnist()
	return np.transpose(mnist['data']), np.transpose(np.ravel(mnist['label']))
	
def get_batch(X, y, batch, batch_size):
	low = batch * batch_size
	high = low + batch_size
	return X[low:high, :], y[low:high]
	
# split data into training & testing sets
def split_train_test():
	X, y = split_X_y()
	return X[:TRAIN_SIZE], y[:TRAIN_SIZE], X[TRAIN_SIZE:], y[TRAIN_SIZE:]

# calls split_train_test, then shuffles training data
def shuffle_train_test():
	trainX, trainY, testX, testY = split_train_test()
	np.random.seed(42)
	train_shuffle = np.random.permutation(TRAIN_SIZE)
	return trainX[train_shuffle], trainY[train_shuffle], testX, testY

# show the ith instance as an image	
def imshow(i):
	X, y = split_X_y()
	dig = X[i]
	dig_img = dig.reshape(28, 28)
	plt.imshow(dig_img, cmap=matplotlib.cm.binary,
				interpolation='nearest')
	plt.axis('off')
	plt.show()
	print(y[i])
			