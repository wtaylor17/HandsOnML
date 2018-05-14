from keras.models import load_model
from mlp_example import split_X_y

recurrent_model_path = 'models/MLPClassifier.h5'

model = load_model(recurrent_model_path)

print('Splitting data...')

X, y = split_X_y()
# reshape with time_steps=1
X = X.reshape((X.shape[0], 1, X.shape[1]))
y = y.reshape((y.shape[0], 1, y.shape[1]))

print(model.evaluate(x=X, y=y))
