import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'

MODEL_PATH = 'models/housing'

EXTRA_ATTRIBUTES = ['rms_per_hhold', 'pop_per_hhold', 'bdrms_per_rm']

# function to fetch data from the url, and put the contents of the .tgz in the path
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	if os.path.isdir(housing_path):
		return
	os.makedirs(housing_path)
	tgz_path = os.path.join(housing_path, 'housing.tgz')
	urllib.request.urlretrieve(housing_url, tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()
	

# loads the data from the path into a DataFrame
def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, 'housing.csv')
	return pd.read_csv(csv_path)
	

# split the data set into 2 random disjoint subsets
def split_train_test(data, test_ratio):
	N = len(data)
	np.random.seed(42)# only want to see a bit of that set!
	shuffled_indices = np.random.permutation(N)
	test_set_size = int(N * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]
	
# perform 'K-fold cross validation' on model
# serialize the model and write it to disk
def cv_score_write(model, X=None, y=None, k=10, name='new_model', update=False, score=True):
	if X is None or y is None:
		cd = clean_data()# default is a k-fold on train_X, train_Y
		X, y = cd[0], cd[1]
	print('\n\n******' + name + '******')
	if score:
		scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=k)
		scores = np.sqrt(-scores)
		print('Scores:', scores)
		print('Mean:', scores.mean())
		print('Standard Deviation:', scores.std())
	path =  MODEL_PATH + '/' +  name + '.pkl'
	if not os.path.isdir(MODEL_PATH):
		os.makedirs(MODEL_PATH)
	if update:
		if os.path.isfile(path):
			os.remove(path)
		f = open(path, 'w')
		f.close()
		joblib.dump(model, path)
		print(model, 'Dumped to', path)
		

# returns a dict of files at the specified path, with names as keys
def load_models(path=MODEL_PATH):
	return {str(f): joblib.load(path + '/' + str(f)) for f in os.listdir(path)}
	

# plot the data geographically
def plot_data_geo(data):
	data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, 
			s=data['population']/100, label='population', c='median_house_value',
			cmap = plt.get_cmap('jet'), colorbar=True)
	plt.show()
	

# outer scope vars used in CombinedAttributesAdder
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
	
# adds the combined attributes to the data frame
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
		return np.c_[X, rooms_per_household, population_per_household,
						bedrooms_per_room]


# returns just the specified attributes from the data frame
class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		return X[self.attribute_names].values

		
# class to binarize labels inside a pipeline
class PipeLineBinarizer(BaseEstimator, TransformerMixin):
	def __init__(self, cat_names=['ocean_proximity']):
		self.sel = DataFrameSelector(cat_names)
	
	def fit(self, X, y=None):	
		return self
		
	def transform(self, X, y=None):
		return LabelBinarizer().fit_transform(self.sel.fit_transform(X))
		

def clean_data(frame=False, housing=None, train=None, test=None):
	# fetch & load data
	fetch_housing_data()
	if housing is None:
		if train is not None and test is not None:
			train_set, test_set = train, test
		else:
			housing = load_housing_data()
			# split into training & testing
			train_set, test_set = split_train_test(housing, 0.2)
	else:
		train_set, test_set = split_train_test(housing, 0.2)
	
	# split data into X (attributes) and Y (labels) components
	train_X = train_set.drop('median_house_value', axis=1)
	train_Y = train_set['median_house_value']
	
	test_X = test_set.drop('median_house_value', axis=1)
	test_Y = test_set['median_house_value']
	
	# drop categorical attribute to get just numerical attributes
	num_attributes = get_numerical_labels() # load numerical attributes into list
	cat_attributes = ['ocean_proximity'] # list of categorical attributes
	
	# pipeline for numerical data
	num_pipeline = Pipeline([
		('selector', DataFrameSelector(num_attributes)),
		('imputer', Imputer(strategy='median')),
		('attribs_adder', CombinedAttributesAdder()),
		('std_scaler', StandardScaler())
	])
	
	# pipeline that does it all
	data_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline), 
						('cat_binarizer', PipeLineBinarizer(cat_attributes))])
	
	# run data through full pipeline
	if frame:
		cols = num_attributes + EXTRA_ATTRIBUTES + get_categorical_labels()
		train_X = pd.DataFrame(data_pipeline.fit_transform(train_X),
						columns=cols)
		test_X = pd.DataFrame(data_pipeline.fit_transform(test_X),
						columns=cols)
	else:
		train_X = data_pipeline.fit_transform(train_X)
		test_X = data_pipeline.fit_transform(test_X)
	return train_X, train_Y, test_X, test_Y
		
def create_new_models():
	train_X, train_Y, test_X, test_Y = clean_data()
	cv_score_write(Ridge(), name='lsq', update=True)
	cv_score_write(KNeighborsRegressor(), name='KNN', update=True)
	cv_score_write(RandomForestRegressor(), name='rfr', update=True)

def grid_search(model, param_grid, k=5, 
					s='neg_mean_squared_error', train_X=None, train_Y=None):
	if train_X is None or train_Y is None:
		cd = clean_data()
		train_X = cd[0]
		train_Y = cd[1]
	gs = GridSearchCV(model, param_grid, cv=k, scoring=s)
	gs.fit(train_X, train_Y)
	return gs.best_estimator_
	
def get_categorical_labels(cat_names=['ocean_proximity']):
	data = load_housing_data()[cat_names].values
	bnz = LabelBinarizer()
	bnz.fit_transform(data)
	return list(bnz.classes_)
	
def get_numerical_labels():
	data = load_housing_data().drop('ocean_proximity', axis=1)
	data = data.drop('median_house_value', axis=1)
	return list(data)
	
def get_all_labels():
	return get_numerical_labels() + EXTRA_ATTRIBUTES + get_categorical_labels()
	
# models = load_models()

# for key in models.keys():
#	cv_score_write(models.get(key), name=key)
