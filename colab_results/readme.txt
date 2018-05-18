The plots in this folder are from a Variational Autoencoder with recurrent LSTM layers.

The data given was a set of time series data, coming from 4 channels. A training instance then has shape (1, 100, 4).
Two thousand training instances were given to the network, the training data then has shape (2000, 100, 4).

The plots are given in folders of 4 images, each image representing
a channel fed into the network. Expected channels are plotted in red, and actual channels are plotted in green.