import keras
import numpy as np
import pandas as pd

df = pd.read_csv('HousingPrices.csv')
df.head()

X = df.drop(columns=['SalePrice'])
Y = df[['SalePrice']]

model = keras.Sequential()
model.add(keras.layers.Dense(8, activation='relu', input_shape=(8,)))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(X,Y, epochs=30, callbacks=[keras.callbacks.EarlyStopping(patience=5)])

test_data = np.array([2003, 854,1710, 2, 1, 3 , 8, 2008])


print(model.predict(test_data.reshape(1,8), batch_size=1))

model.save('housing_model.h5')

old_model = keras.models.load_model('housing_model.h5')