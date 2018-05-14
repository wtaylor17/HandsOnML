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
	
"""
End of code from Part1/mnist.py
"""

class DNNPlain:
	def __init__(self, n_inputs=28*28, n_outputs=10, learning_rate=0.01, path='deep_nn/saves'):
		self.X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
		self.y = tf.placeholder(tf.float32, shape=(None), name='y')
		self.n_outputs = 10
		self.learning_rate = learning_rate
		self.layers = list()
		self.training_op = None
		self.accuracy = None
		self.save_path = path
		
	def create_layers(self, layer_sizes, layer_names, n_layers, activation='relu'):
		if len(layer_sizes) != n_layers:
			raise AttributeError('Inconsistent dimensions: layer_sizes and n_layers')
		with tf.name_scope('dnn'):
			self.layers.append(tf.fully_connected(X, layer_sizes[0], scope=layer_names[0]))
			for i in range(1, n_layers - 1):
				self.layers.append(tf.fully_connected(self.layers[i - 1], layer_sizes[i], scope=layer_names[i]))
			self.layers.append(tf.fully_connected(self.layers[-1], layer_sizes[-1], scope=layer_names[-1], activation_fn=None))
			with tf.name_scope('loss'):
				xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.layers[-1])
				loss = tf.reduce_mean(xentropy, name='loss')
				with tf.name_scope('train'):
					optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
					self.training_op = optimizer.minimize(loss)
			with tf.name_scope('eval'):
				correct = tf.nn.in_top_k(self.layers[-1], y, 1)
				self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	
	def mini_batch_train(n_epochs=400, batch_size=50, data=split_train_test()):
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		with tf.Session() as sess:
			file_writer = tf.summary.FileWriter(self.save_path, sess.graph)
			for epoch in range(n_epochs):
				tf.run(init)
				for iteration in int(TRAIN_SIZE / batch_size):
					batch = get_batch(data[0], data[1], iteration, batch_size)
					sess.run(self.training_op, feed_dict={self.X: batch[0], self.y: batch[1]})
				acc_train = self.accuracy.eval(feed_dict={self.X: batch[0], self.y: batch[1]})
				print(epoch, 'Training accuracy', acc_train)
			saver.save(sess, path + '/deepNN.ckpt')
			file_writer.close()
			