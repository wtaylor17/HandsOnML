from sklearn.neural_network import MLPClassifier
import mnist
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.metrics import precision_score

mlp = MLPClassifier(random_state=42)
X, y = mnist.split_X_y()
y = y.reshape(y.shape[0], 1)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape, y_train.shape)
mlp.fit(X_train, np.ravel(y_train))
pred_y = mlp.predict(X_test)
print(precision_score(np.ravel(y_test), pred_y, average='micro'))
