import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.neighbors import KNeighborsClassifier 
import os
import numpy as np
import housing

MODEL_PATH = 'models/mnist'
FILE_PATH = MODEL_PATH + '/knc.pkl'
s_op = 'neg_mean_squared_error'

trainX, trainY, testX, testY = mnist.shuffle_train_test()
print('data loaded.')

knc = KNeighborsClassifier()
knc.fit(testX, testY)
my_dict = {'model': knc, 'train_data': trainX, 'train_label': trainY}
print('model loaded.')
pred_y = knc.predict(testX)
print('prediction made.')
print(precision_score(testY, pred_y, average='micro'))
