import numpy as np
import tensorflow as tf

MODEL_ROOT = 'models/tf_intro/'
LOG_ROOT = 'logs/tf_intro/'

BATCH_PATH = MODEL_ROOT + 'batch/'
BATCH_PATH_LOG = LOG_ROOT + 'batch'

STOCHASTIC_PATH = MODEL_ROOT + 'stochastic/'
STOCHASTIC_PATH_LOG = LOG_ROOT + 'stochastic'

# X, y are numpy arrays in all cases

def batch_gd(X, y, n_epochs=100, eta=0.0001, save=True):
	print('Creating tensors...')
	m, n = X.shape
	X = tf.constant(X, dtype=tf.float32, name='X')
	y = tf.constant(y, dtype=tf.float32, name='y')
	theta = tf.Variable(tf.random_uniform([n, 1], -1, 1), dtype=tf.float32, name='theta')
	print('Done.\nComputing gradient stuff...')
	y_pred = tf.matmul(X, theta, name='predictions')
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error), name='mse')
	gradients = tf.gradients(mse, [theta])[0]
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	training_op = tf.assign(theta, theta - eta * gradients)
	mse_summary = tf.summary.scalar('MSE', mse)
	print('Done.')
	last_err = 1000000000000
	with tf.Session() as sess:
		file_writer = tf.summary.FileWriter(BATCH_PATH_LOG, sess.graph)
		print('Launching batch GD.')
		for epoch in range(n_epochs):
			sess.run(init)
			sess.run(training_op)
			current_err = np.sqrt(mse.eval())
			print('Current err:', current_err)
			summary_str = mse_summary.eval()
			file_writer.add_summary(summary_str, epoch)
			"""if current_err < last_err:
				eta *= 0.8
				print('Decreasing learning rate', eta)
				training_op = tf.assign(theta, theta - eta * gradients)
			last_err = current_err"""
			if current_err < 40000:
				print('TRAINING CONVERGED: ', current_err)
				print(sess.run(theta))
				if save:
					saver.save(sess, BATCH_PATH + 'linreg.ckpt')
					file_writer.close()
				sess.close()
				quit()


def stochastic_gd(X, y, n_epochs=100, eta=0.000001, save=True):
	print('Creating tensors...')
	m, n = X.shape
	n_epochs *= m
	np.random.seed(42)
	X_ph = tf.placeholder(tf.float32, shape=(1, n), name='X_i')
	y_ph = tf.placeholder(tf.float32, shape=(1, 1), name='y_i')
	theta = tf.Variable(tf.random_uniform([n, 1], -1, 1), dtype=tf.float32, name='theta')
	print('Done.\nComputing gradient stuff...')
	y_pred = tf.matmul(X_ph, theta, name='predictions')
	error = y_pred - y_ph
	mse = tf.reduce_mean(tf.square(error), name='mse')
	gradients = tf.gradients(mse, [theta])[0]
	init = tf.global_variables_initializer()
	training_op = tf.assign(theta, theta - eta * gradients)
	saver = tf.train.Saver()
	mse_summary = tf.summary.scalar('MSE', mse)
	print('Done.')
	last_m = 0
	with tf.Session() as sess:
		file_writer = tf.summary.FileWriter(STOCHASTIC_PATH_LOG, sess.graph)
		print('Launching stochastic GD.')
		for epoch in range(n_epochs):
			sess.run(init)
			i = np.random.randint(0, m)
			X_i = X[i, :].reshape((1, 9))
			y_i = y[i].reshape((1, 1))
			sess.run(training_op, feed_dict={X_ph: X_i, y_ph: y_i})
			current_err = np.sqrt(mse.eval({X_ph: X_i, y_ph: y_i}))
			last_m += current_err
			summary_str = mse_summary.eval({X_ph: X_i, y_ph: y_i})
			file_writer.add_summary(summary_str, epoch)
			if (epoch + 1) % 1000 == 0:
				last_m /= 1000
				print('Average of last', 1000, 'instances:', last_m)
				if last_m < 40000:
					print('TRAINING CONVERGED: ', current_err)
					print(sess.run(theta))
					if save:
						saver.save(sess, STOCHASTIC_PATH + 'linreg.ckpt')
						file_writer.close()	
					sess.close()
					quit()
				last_m = 0