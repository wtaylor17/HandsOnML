import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error as mse
import housing_script as hs
import housing as h

FINAL_MODEL_PATH = 'models/final/'
PARAM_GRID = hs.param_grid

def new_fitted_iter():
	data = hs.get_iter_data()
	rfr_ = RandomForestRegressor()
	rfr_.fit(data[0], data[1])
	return rfr_

def new_fitted_clean():
	data = h.clean_data()
	rfr = RandomForestRegressor()
	rfr.fit(data[0], data[1])
	return rfr
	
def grid_search_iter():
	data = hs.get_iter_data()
	return h.grid_search(RandomForestRegressor(),
			PARAM_GRID, train_X=data[0], train_Y=data[1])
			
def grid_search_clean():
	return h.grid_search(RandomForestRegressor(), PARAM_GRID)
	
def write_model(model_, name='new.pkl'):
	if not os.path.isdir(FINAL_MODEL_PATH[:-1]):
		os.makedirs(FINAL_MODEL_PATH[:-1])
	path = FINAL_MODEL_PATH + name
	if os.path.isfile(path):
		os.remove(path)
	f = open(path, 'w')
	f.close()
	joblib.dump(model_, path)
	print(model_, 'Dumped to', path)
	
def create_all():
	models_ = list()
	models_.append(new_fitted_iter())
	print('APPENDED 0')
	models_.append(grid_search_iter())
	print('APPENDED 1')
	models_.append(new_fitted_clean())
	print('APPENDED 2')
	models_.append(grid_search_clean())
	print('MODELS CREATED')
	return models_
	
def test(model_, data=hs.get_iter_data(), name='RFR'):
	print('******', name, '*******')
	y_pred = model_.predict(data[2])
	print('MSE:', mse(data[3], y_pred))
	print('NORM:', np.sqrt(mse(data[3], y_pred)))
	
models = create_all()
clean = h.clean_data()

test(models[0], name='iter')
test(models[1], name='iter_gs')
test(models[2], data=clean, name='clean')
test(models[3], data=clean, name='clean_gs')
