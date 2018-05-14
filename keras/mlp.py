from keras import Sequential
from keras.layers import Dense, Dropout, Layer
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import to_categorical, plot_model
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.layers import GRU
import numpy as np


class NNClassifier:

	"""
	Class that encapsulates a multilayer perceptron classifier
	using the keras library in order to visualize the system
	more effectively.
	
	Written by William Taylor-Melanson on 2018-05-14
	"""
	
	def __init__(self,
				logdir='logs',
				board=True,
				histogram_freq=5,
				save_on_exit=True,
				save_freq=-1,
				plot_model_on_compile=True,
				show_shapes=False,
				png_path='images/MLPClassifier.png',
				model_path='models/MLPClassifier.h5'):
				
		"""
		__init__
		
		Arguments:
		
		logdir - the directory to which the tensorboard events file is written.
		
		board - true if a tensorboard callback should be created.
		
		histogram_freq - the frequency (in epochs) with which the TB callback is called.
		
		save_freq - the frequency (in epochs) with which the model should be written to disk.
		
		save_on_exit - true if the model should be written to file upon training completion.
		
		plot_model_on_compile - true if the MLP should be plotted upon a call to compile().
		
		show_shapes - true if the shapes of the tensors passed are shown in the model plot.
		
		png_path - the file path to which the plotted model is written.
		
		model_path - path to save/load the model to and from.
		
		Public feilds are layer_names (names of layers in MLP) and model (Sequential object)
		"""
		
		self.load = load
		self.__callbacks = list()
		
		if save_on_exit:
			self.model_path = model_path
		else:
			self.model_path = None
			
		self.model = Sequential()
		self.__n_layers = 0
		self.layer_names = list()
		if board:
			# create tensorboard callback
			self.__callbacks.append(TensorBoard(logdir,
									histogram_freq=histogram_freq))
		
		if save_freq > 0:
			if self.model_path is None:
				raise ValueError('model_path cannot be none if save_freq > 0.')
			self.__callbacks.append(ModelCheckpoint(model_path, period=save_freq))
		
		if plot_model_on_compile:
			self.__plot_args= png_path, show_shapes
		else:
			self.__plot_args = None
	
	
	def add_dense_layer(self, 
						output_dim, 
						input_dim=None, 
						activation='relu',
						name=None):
						
		"""
		add_dense_layer
		
		Adds a dense layer of neurons to this MLP.
		
		Arguments:
		
		output_dim - the dimension of the output tensor for this layer.
		
		input_dim - the dimension of the input tensor for this layer, only needs
		to be specified for the initial layer.
		
		activation - the activation function for this layer, type str or callable.
		
		Raises ValueError if input_dim is None and n_layers is 0.
		"""
		
		layer_name = name if name is not None else 'layer_' + str(self.__n_layers)
		if input_dim is None:
			if self.__n_layers is 0:
				raise ValueError('Argument input_dim must be provided for first layer')
			else:
				self.model.add(Dense(output_dim,
									activation=activation,
									name=layer_name))
				self.__n_layers += 1
				self.layer_names.append(layer_name)
		else:
			self.model.add(Dense(output_dim,
								input_dim=input_dim,
								activation=activation,
								name=layer_name))
			self.__n_layers += 1
			self.layer_names.append(layer_name)
			
	
	def add(self, layer):
		"""
		add
		
		Adds the specified layer to self.model.
		
		front-end use of Sequential.add
		see keras documentation for details
		"""
		
		layer_name = 'layer_' + str(self.__n_layers)
		self.layer_names.append(layer_name)
		self.model.add(layer)
		self.__n_layers += 1
		


	def add_output_layer(self,
						output_dim,
						activation='softmax',
						compile=False,
						loss='categorical_crossentropy',
						metrics=[categorical_accuracy],
						optimizer='rmsprop',
						name=None):
	
		"""
		add_output_layer
		
		Adds the output layer to this model, and possibly compiles.
		
		Arguments:
		
		output_dim - the dimension of the output tensor for this layer.
		
		activation - the activation function for this layer, str or callable.
		
		compile - true if this model is compiled after the layer is added.
		
		loss, metrics, and optimizer - see NNClassifier.compile().
		"""
		
		layer_name = name if name is not None else 'output_layer'
		self.model.add(Dense(output_dim,
							activation=activation,
							name=layer_name))
		self.__n_layers += 1
		self.layer_names.append(layer_name)
							
		if compile:
			self.compile(optimizer=optimizer,
						loss=loss,
						metrics=metrics)
						
	
	def add_dropout_layer(self,
						ratio,
						random_seed=42,
						name=None):
						
		"""
		add_dropout_layer
		
		Adds a Dropout to the MLP to prevent overfitting.
		
		Arguments:
		
		ratio - the ratio of random inputs to drop.
		
		random_seed - seed for reproducing results.
		
		name - the name of this layer, will be 'dropout_' + str(ratio)
				if not specified.
		"""
		
		layer_name = name if name is not None else 'dropout_' + str(ratio)
		
		if self.__n_layers is 0:
			raise IndexError('Cannot add dropout layer, number of layers is zero.')
		else:
			self.model.add(Dropout(ratio, seed=random_seed))
			self.__n_layers += 1
			self.layer_names.append(layer_name)
			
		
	def add_recurrent_layer(self,
							units,
							activation='tanh',
							recurrent_activation='hard_sigmoid',
							implementation=2,
							stateful=False,
							input_dim=None,
							name=None):
							
		"""
		add_recurrent_layer
		
		Adds a RNN layer to this model.
		
		Arguments:
		
		units - dimension of output space for this layer.
		
		activation - activation function, str or callable.
		
		recurrent_activation - activation function for recurrent step.
		
		implementation - 1 for several small computations, 2 for big ones but not as many.
		
		stateful - if true, last state for sample used as first state on next batch.
		
		input_dim - only needed if this is the first layer.
		
		name - name of this layer.
		"""
		
		layer_name = name if name is not None else 'recurrent_layer_' + str(self.__n_layers)
		
		recurrent_layer = GRU(units,
							activation=activation,
							recurrent_activation=recurrent_activation,
							implementation=implementation,
							stateful=stateful,
							input_dim=input_dim,
							return_sequences=True)
		self.model.add(recurrent_layer)
							
		self.__n_layers += 1
		self.layer_names.append(layer_name)
		
						
						
	def compile(self,
				optimizer='rmsprop', 
				loss='categorical_crossentropy',
				metrics=[categorical_accuracy]):
		
		"""
		compile
		
		Compiles this model for use.
		
		Arguments:
		
		optimizer - the optimizer for use in training, str or callable.
		
		loss - the loss function for this model, str or callable.
		
		metrics - a list of metrics (str or callable) for this model.
		"""
		
		self.model.compile(optimizer=optimizer,
					loss=loss,
					metrics=metrics)
		if self.__plot_args is not None:
			plot_model(self.model, self.__plot_args[0], show_shapes=self.__plot_args[1])
			
			
	def fit(self, 
			x=None,
			y=None,
			batch_size=None,
			epochs=20,
			verbose=1,
			validation_split=0.2,
			validation_data=None,
			initial_epoch=1):
		
		"""
		fit
		
		Fits data to this model
		
		front-end use of Sequential.fit
		see keras documentation for details
		
		"""
		
		if not self.__callbacks:
			callbacks = None
		else:
			callbacks = self.__callbacks
		
		self.model.fit(x=x,
					y=y,
					batch_size=batch_size,
					epochs=epochs,
					verbose=verbose,
					callbacks=callbacks,
					validation_split=validation_split,
					validation_data=validation_data,
					initial_epoch=initial_epoch)
					
		if self.model_path is not None:
			self.model.save(self.model_path)
			
		
	def evaluate(self, 
				x=None, 
				y=None, 
				batch_size=None, 
				verbose=1):
				
		"""
		
		front-end use of Sequential.evaluate
		see keras documentation for details
		
		"""
		
		return self.model.evaluate(x=x,
								y=y,
								batch_size=batch_size,
								verbose=verbose)
								
	
	def predict(self,
				x,
				batch_size=None,
				verbose=0):
		
		"""
		
		front-end use of Sequential.predict
		see keras documentation for details
		
		"""
		
		return self.model.predict(x,
								batch_size=batch_size,
								verbose=verbose)
	
	def save(self, path=None):
	
		"""
		save
		
		saves self.model to the specified path, or the default path 
		self.model_path if no path is specified.
		
		"""
		
		path = path if path is not None else self.model_path
		self.model.save(path)
