import numpy as np
from sklearn.datasets import fetch_california_housing
from linreg_gd import batch_gd, stochastic_gd
import tensorflow as tf

MODEL_ROOT = 'models/tf_intro/'
LOG_ROOT = 'logs/tf_intro/'

BATCH_PATH = MODEL_ROOT + 'batch/'
BATCH_PATH_LOG = LOG_ROOT + 'batch/test'

STOCHASTIC_PATH = MODEL_ROOT + 'stochastic/'
STOCHASTIC_PATH_LOG = LOG_ROOT + 'stochastic'

def shuffle(data, target, ratio=0.8):
	m = data.shape[0]
	np.random.seed(42)
	shuffled_indices = [i for i in range(m)]
	np.random.shuffle(shuffled_indices)
	test_indices = shuffled_indices[int(ratio * m):]
	train_indices = shuffled_indices[:int(ratio * m)]
	return data[train_indices], target[train_indices], data[test_indices], target[test_indices]

"""
Linear regression using gradient descent
"""

def perform_batch_gd():
	housing = fetch_california_housing()
	m, n = housing.data.shape
	housing_data_biased = np.c_[np.ones((m, 1)), housing.data]
	trainX, trainY, testX, testY = shuffle(housing_data_biased, housing.target)
	batch_gd(trainX, trainY)

def test_gd(path='batch'):
	housing = fetch_california_housing()
	m, n = housing.data.shape
	housing_data_biased = np.c_[np.ones((m, 1)), housing.data]
	data = shuffle(housing_data_biased, housing.target)
	
	X = tf.constant(data[2], name='X', dtype=tf.float32)
	y = tf.constant(data[3], name='y', dtype=tf.float32)
	
	theta = tf.Variable(tf.random_uniform([n + 1, 1], -1, 1), dtype=tf.float32, name='theta')
	saver = tf.train.Saver({'theta': theta})
	
	y_pred = tf.matmul(X, theta, name='predictions')
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error), name='mse')
	
	with tf.Session() as sess:
		saver.restore(sess, MODEL_ROOT + path + '/linreg.ckpt')
		file_writer = tf.summary.FileWriter(LOG_ROOT + path + '/test', sess.graph)
		
		print('ERROR OF PREDICTIONS:', np.sqrt(sess.run(mse)))
		file_writer.close()
		
		
def perform_stochastic_gd():
	housing = fetch_california_housing()
	m, n = housing.data.shape
	housing_data_biased = np.c_[np.ones((m, 1)), housing.data]
	trainX, trainY, testX, testY = shuffle(housing_data_biased, housing.target)
	stochastic_gd(trainX, trainY)
