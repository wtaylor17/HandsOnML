import housing
from sklearn.metrics import mean_squared_error as mse
import numpy as np

"""
functions to improve/test accuracy of the chosen model
"""

r_attribs = ['ISLAND', 'NEAR BAY', 'NEAR OCEAN', 'households', '<1H OCEAN',
									'total_bedrooms', 'population', 'total_rooms']
param_grid = [{'n_estimators': [30], 'max_features': [2, 4, 6, 8], 'max_depth': [33, 22, 44]}]


# remove all rows such that at least one of the attributes in the row
# is more that 2.5 std's from the mean of that attribute
def remove_outliers(data):
	cols = list(data.columns.values)
	if 'ocean_proximity' in cols:
		cols.remove('ocean_proximity')
	means = {c: data[c].mean() for c in cols}
	stdevs = {c: data[c].std() for c in cols}
	for c in cols:
		data = data[np.absolute(getattr(data, c) - means[c]) < 2.5 * stdevs[c]]
	return data

# performs a grid search and k-fold cv
def grid_search():
	rfr = housing.load_models()['rfr.pkl']
	housing.cv_score_write(housing.grid_search(rfr, param_grid),
							name='rfr', update=True)
							

# fetches the version of the data set being currently used on the rfr
def get_iter_data():
	g_set = housing.load_housing_data()
	training, testing = housing.split_train_test(g_set, 0.2)
	training = remove_outliers(training)
	cd = housing.clean_data(frame=True, train=training, test=testing)
	return cd[0].drop(r_attribs, axis=1), cd[1], \
			cd[2].drop(r_attribs, axis=1), cd[3]


# test on the test data, show the error			
def test_rfr(path='rfr.pkl', cd=get_iter_data()):
	rfr = housing.load_models()[path]
	predictions = rfr.predict(cd[2].values)
	err = mse(cd[3], predictions)
	print(np.sqrt(err))
	

# iterate the rfr to (hopefully) make it better.
def iter_rfr(attr=r_attribs, path='rfr.pkl'):
	cd = get_iter_data() # get the data valid on this iteration
	X = cd[0]
	y = cd[1]
	rfr = housing.load_models()[path]
	# perform a grid search and write the optimal model to disk
	housing.cv_score_write(housing.grid_search(rfr, param_grid, train_X=X, train_Y=y),
							name=path[:-4], update=True, score=False)


# measures the importance of the attributes with respect to the rfr
def rfr_importances(path='rfr.pkl', removal_attr=[]):
	rfr = housing.load_models()[path]
	ftr_imp = rfr.feature_importances_
	attributes = housing.get_all_labels()
	if removal_attr:
		for a in removal_attr:
			if a in attributes:
				attributes.remove(a)
	return sorted(zip(ftr_imp, attributes), reverse=True)
