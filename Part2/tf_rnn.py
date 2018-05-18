import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn import BasicRNNCell, OutputProjectionWrapper
import numpy as np
from matplotlib import pyplot as plt

func = lambda x: x * np.sin(x) / 3 + 2 * np.sin(5 * x)

t_vals = [[[t] for t in np.arange(n, n + 2, 0.1)] for n in range(0, 30, 2)]
series = [[[func(slice) for slice in col] for col in row] for row in t_vals]
LOG_PATH = 'logs/time_series'

def get_batch(batch, size=5):
	low = (batch * size) % (40 - size)
	high = low + size
	return t_vals[low:high], series[low:high]
	
n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = OutputProjectionWrapper(
				BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
				output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y), name='loss')
loss_summary = tf.summary.scalar('loss', loss)

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

batch_size = 100
n_iterations = 20000

with tf.Session() as sess:
	init.run()
	writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
	for iteration in range(n_iterations):
		X_batch, y_batch = t_vals, series
		sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		if iteration % 100 == 0:
			summary_str = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
			writer.add_summary(summary_str, iteration)
			print(iteration, 'Loss:', loss.eval(feed_dict={X: X_batch, y: y_batch}))
		
	test_X, test_y = t_vals, series
	
	pred_y = sess.run(outputs, feed_dict={X: test_X})
	
	plt.plot(np.ravel(test_X), np.ravel(test_y), 'g--', np.ravel(test_X), np.ravel(pred_y), 'r--')
	plt.show()
	writer.close()