from housing import load_housing_data, clean_data
from housing_script import remove_outliers
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import numpy as np

def train_write_svr(model=SVR(kernel=linear()), score=True, k=3):
	housing_data = clean_data(frame=True, housing=remove_outliers(load_housing_data()))
	model.fit(housing_data[0], housing_data[1])
	if score:
		scores = cross_val_score(model, housing_data[0], housing_data[1], scoring='neg_mean_squared_error', cv=k)
		scores = np.sqrt(-scores)
		print('Scores:', scores)
		print('Mean:', scores.mean())
		print('Standard Deviation:', scores.std())
