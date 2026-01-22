import tensorflow as neuralNet

mnist = neuralNet.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()

model = neuralNet.keras.models.load_model('handwritten.keras')
x_test = neuralNet.keras.utils.normalize(x_test, axis=1)
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)