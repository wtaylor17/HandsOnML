# -*- coding: utf-8 -*-
"""Copy of time_series.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a60bSAL0vPinA_Ds2gx8eC2yQVSV-sbs
"""

#@title Data creation
import numpy as np
n_instances = 2000 # 2,000 training instances
n_steps = 100 # 100 time steps per instance
n_channels = 4 # 4 channels

# input data is of shape (n_instances, n_steps, n_channels)

def channel(t):
  c0 = np.sin(3 * t) ** 2
  c1 = np.sin(4 * t) ** 2
  c2 = np.cos(4 * t) ** 2
  c3 = np.sin(2 * t) ** 2
  return [c0, c1, c2, c3]

data = [[channel((t * 0.01) + np.random.randint(low=0, high=5))
         for t in range(n_steps)]
        for k in range(n_instances)]

data = np.array(data).astype('float32')

#@title sample function
# helper function to comput z = mu + e * std
def sample(args):
  mean_, log_sig_ = args
  epsilon = K.random_normal(shape=(batch_size, n_steps, n_latent))
  return mean_ + epsilon * K.exp(log_sig_)

#@title imports and vars
import keras
from keras import backend as K
from keras.layers import Dense, LSTM, Lambda
from keras.engine import Input
from keras.models import Model
from keras import losses, optimizers
import tensorflow as tf

K.clear_session()

n_hidden1 = 250
n_hidden2 = 500
n_latent = 1
batch_size = 100
n_epochs = 50

#@title encoder creation
x = Input(shape=(n_steps, n_channels,), name='inputs')

hidden1 = Dense(n_hidden1, activation='tanh', name='x')(x)
hidden2 = LSTM(n_hidden2, activation='tanh',
              return_sequences=True)(hidden1)
z_mean = Dense(n_latent, name='mean')(hidden2)
z_log_sigma = Dense(n_latent, name='sigma')(hidden2)

z = Lambda(sample, name='z')([z_mean, z_log_sigma])

encoder = Model(x, z)

#@title decoder creation
latent_inputs = Input(shape=(n_steps, n_latent,), name='lat_in')
hidden3 = LSTM(n_hidden2, return_sequences=True,
               activation='tanh')(latent_inputs)

outputs = Dense(n_channels, activation='sigmoid',
                name='outputs')(hidden3)

decoder = Model(inputs=latent_inputs, outputs=outputs)
# decoder.summary()

#@title end-to-end model creation
y = encoder(x)
outputs = decoder(y)

model = Model(inputs=x, outputs=outputs)

# reconstruction_loss = losses.binary_crossentropy(x, outputs)
# 
# kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
# print(kl_loss.shape)
# kl_loss = K.sum(kl_loss, axis=2)
# kl_loss *= -0.5 * 10

# total_loss = K.mean(reconstruction_loss + kl_loss)
# model.add_loss(total_loss)
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x=data, y=data, epochs=50, batch_size=batch_size)

batch = data[:100]

x_pred = model.predict(batch, batch_size=batch_size) 
# shape of prediction data and batch are both (100, 100, 4)

t_1 = batch[1]
t_2 = x_pred[1]


import matplotlib.pyplot as plt
for j in range(n_channels):
  plt.plot(t_1[:, j], 'g', t_2[:, j], 'r')
  plt.show()