import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
The following code builds a simple linear
autoencoder to perform PCA on a 3D datset,
projecting it to 2d.
"""

n_inputs = 3
n_hidden = 2
n_ouputs = n_inputs

learning_rate = 0.001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=None)
outputs = fully_connected(hidden, n_ouputs, activation_fn=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

X_train, X_test = np.random.rand(300, 3), np.random.rand(60, 3)

n_iterations = 1000
codings = outputs

with tf.Session() as sess:
	init.run()
	for iteration in range(n_iterations):
		training_op.run(feed_dict={X: X_train})
	codings_val = codings.eval(feed_dict={X: X_test})
	fig1 = plt.figure()
	fig2 = plt.figure()
	ax_test = Axes3D(fig1)
	ax_test.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], label='test_data')
	ax_pred = fig2.add_subplot(1, 1, 1)
	ax_pred.scatter(codings_val[:, 0], codings_val[:, 1])
	plt.show()
