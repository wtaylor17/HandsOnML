from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
					metrics=['accuracy'])
model.fit(data, labels, epochs=10, batch_size=32)

testX = np.random.random((100, 100))
testY = np.random.randint(2, size=(100, 1))

pred_y = model.predict(testX)
print(sum(np.round(pred_y) == testY))
