import mlp_example as helper
from mlp import NNClassifier as CLS
from keras.layers import InputLayer

print('Splitting data...')

X, y = helper.split_X_y()
# reshape with time_steps=1
X = X.reshape((X.shape[0], 1, X.shape[1]))
y = y.reshape((y.shape[0], 1, y.shape[1]))

print('Done.\nCreating model..')

cls = CLS(logdir='logs/recurrent', 
		show_shapes=True,
		save_freq=2)
		
print('Done\nadding first layer...')
cls.add_recurrent_layer(300, input_dim=28*28)
print('Done\nadding second layer...')
cls.add_recurrent_layer(100)
print('Done\nadding output layer...')
cls.add_output_layer(10, compile=True)

print('Done.\n\n\n\n')

try:
	cls.fit(x=X, y=y, verbose=1)
except BaseException:
	cls.save()
