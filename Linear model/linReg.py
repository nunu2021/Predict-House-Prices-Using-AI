import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IowaHousingPrices.csv')

squareFeet = df[['SquareFeet']].values
salePrice = df[['SalePrice']].values

modal = keras.Sequential() # this is what tells the computer that our modal is a linREG model.
modal.add(keras.layers.Dense(1, input_shape=(1,)))
modal.compile(keras.optimizers.Adam(lr=1), 'mean_squared_error')

modal.fit(squareFeet, salePrice, epochs=30, batch_size=10)


df.plot(kind='scatter', x='SquareFeet',y='SalePrice',title='Housing pricing in IOWA')
y_prod = modal.predict(squareFeet)

plt.plot(squareFeet, y_prod, color='red')
plt.show()


newSF = 2000
print(modal.predict([newSF]))

