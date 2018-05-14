from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from keras.utils import to_categorical, plot_model
from keras.metrics import categorical_accuracy
from scipy import io
import numpy as np

""" 
	The following 3 functions are from HandsOnML/Part1/mnist.py,
	with one slight modification for use with the MLP.
	The modification is the call to to_categorical in order
	to format the labels correctly.
"""

DL_PATH = 'datasets/mnist/mnist-original.mat'

# load the data
def load_mnist():
	return io.loadmat(DL_PATH)

# split data into data and label
def split_X_y():
	mnist = load_mnist()
	return np.transpose(mnist['data']), to_categorical(np.transpose(np.ravel(mnist['label'])))
	
# split data into training & testing sets
def split_train_test():
	X, y = split_X_y()
	return X[:60000], y[:60000], X[60000:], y[60000:]

"""
	An example using the keras library to create a multi-layer perceptron
	to classify the mnist (handwritten digits) dataset.


# load mnist data
data = split_train_test()

# create tensorboard callback
tb_callback = TensorBoard('logs/mnist_demo', histogram_freq=5)

# put together network
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=28*28, name='hidden_layer_0'))
model.add(Dense(50, activation='tanh', name='hidden_layer_1'))
model.add(Dense(10, activation='softmax', name='output_layer'))

model.compile(optimizer='rmsprop',
 loss='categorical_crossentropy',
 metrics=[categorical_accuracy])
 
plot_model(model, to_file='mlp_model.png', rankdir='LR')
 
model.fit(x=data[0], y=data[1],
 epochs=100, callbacks=[tb_callback],
 validation_data=(data[2], data[3]))

print(model.evaluate(x=data[2], y=data[3]))
"""




"""
mlp script with class


from mlp import NNClassifier as CLS

print('Splitting data...')

X, y = split_X_y()

print('Done.\nCreating model..')

cls = CLS(logdir='logs/mlp_example', 
		show_shapes=True)

print('Done\nadding first layer...')

cls.add_dense_layer(300, input_dim=28*28)
print('Done\nadding second layer...')
cls.add_dense_layer(100)
print('Done\nadding dropout layer...')
cls.add_dropout_layer(0.3)
print('Done\nadding third layer...')
cls.add_dense_layer(32)
print('Done\nadding output layer...')
cls.add_output_layer(10, compile=True)

print('Done.\n\n\n\n')
try:
	cls.fit(x=X, y=y, verbose=1, epochs=500)
except Exception:
	cls.save()
"""