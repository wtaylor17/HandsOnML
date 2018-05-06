import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'

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
	

# helper function to print scores
def display_scores(scores):
	print('Scores:', scores)
	print('Mean:', scores.mean())
	print('Standard Deviation:', scores.std())
	

def plot_data_geo(data):
	data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, 
			s=data['population']/100, label='population', c='median_house_value',
			cmap = plt.get_cmap('jet'), colorbar=True)
	plt.show()
	

# outer scope vars used in CombinedAttributesAdder
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
	
# adds the combined attributes to the data frame
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room=True):
		self.add_bedrooms_per_room = add_bedrooms_per_room

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household,
						bedrooms_per_room]
		else:
			return np.c_[X, rooms_per_household, population_per_household]


# returns just the specified attributes
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

# fetch & load data
fetch_housing_data()
housing = load_housing_data()

# split into training & testing
train_set, test_set = split_train_test(housing, 0.2)

# split data into X (attributes) and Y (labels) components
train_X = train_set.drop('median_house_value', axis=1)
train_Y = train_set['median_house_value']

test_X = test_set.drop('median_house_value', axis=1)
test_Y = test_set['median_house_value']

# drop categorical attribute to get just numerical attributes
housing_num = train_X.drop('ocean_proximity', axis=1)
num_attributes = list(housing_num) # load numerical attributes into list
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
train_X_prepared = data_pipeline.fit_transform(train_X)
test_X_prepared = data_pipeline.fit_transform(test_X)

# perform K-fold cross validation on DTR
# splits training set into K distinct folds
# trains and evaluates model K times
# picking one fold for validation and the other K-1 for training
# the result is an array containing the K evaluating scores
k = 10
scores = cross_val_score(DecisionTreeRegressor(), train_X_prepared, train_Y,
			scoring='neg_mean_squared_error', cv=k)
display_scores(np.sqrt(-scores))
