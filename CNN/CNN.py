import keras
import matplotlib as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


plt.inshow(x_train[0])

plt.show()

print(y_train[0])
# still working on this
