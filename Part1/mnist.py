import os
from scipy import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

url = 'https://github.com/amplab/datascience-sp14/blob/master/lab7/mldata/mnist-original.mat'
DL_PATH = 'datasets/mnist/mnist-original.mat'

# load the data
def load_mnist():
	return io.loadmat(DL_PATH)

# split data into data and label
def split_X_y():
	mnist = load_mnist()
	return np.transpose(mnist['data']), np.transpose(np.ravel(mnist['label']))
	
# split data into training & testing sets
def split_train_test():
	X, y = split_X_y()
	return X[:60000], y[:60000], X[60000:], y[60000:]

# calls split_train_test, then shuffles training data
def shuffle_train_test():
	trainX, trainY, testX, testY = split_train_test()
	np.random.seed(42)
	train_shuffle = np.random.permutation(60000)
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

# creates a binary classifier to identify the digit 'i'
def create_binary_classifier(i, shuffle=True, model=SGDClassifier(random_state=42, max_iter=5)):
	if shuffle:
		trainX, trainY, testX, testY = shuffle_train_test()
	else:
		trainX, trainY, testX, testY = split_train_test()
	y_train_i = (trainY == [i])
	y_test_i = (testY == [i])
	sgd = clone(model)
	sgd.fit(trainX, y_train_i)
	return {'model': sgd, 'train_data': trainX,
			'test_data': testX, 'train_label': y_train_i,
					'test_label': y_test_i, 'digit': i}

# uses k-folds to cross validate a binary classifier
def cv_score_classifier(cls, k=3, state=42):
	skfolds = StratifiedKFold(n_splits=k, random_state=state)
	model = cls['model']
	X = cls['train_data']
	y = cls['train_label']
	for train_index, test_index in skfolds.split(X, y):
		model_clone = clone(model)
		trainX = X[train_index]
		trainY = (y[train_index])
		testX = X[test_index]
		testY = (y[test_index])
		
		model_clone.fit(trainX, trainY)
		pred = model_clone.predict(testX)
		n_correct = sum(pred == testY)
		print(n_correct / len(pred))
		
# uses cross validation with the specified method 'm'
def cv_predict(cls, k=3, m='predict'):
	return cross_val_predict(cls['model'], cls['train_data'],
			cls['train_label'], cv=k, method=m)

# returns the confusion matrix of the given binary classifier dict
def cv_matrix_binary_classifier(cls, k=3, m='predict'):
	pred = cross_val_predict(cls['model'], cls['train_data'],
			cls['train_label'], cv=k, method=m)
	return confusion_matrix(cls['train_label'], pred)
