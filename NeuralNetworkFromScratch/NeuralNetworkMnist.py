import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# from (6000, 28, 28) to (6000, 784)
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

# from 0-255 to 0-1
X_train: np.ndarray = X_train / 255
X_test: np.ndarray = X_test / 255


def Get_HandWrittenNumbersFromZeroToNine(train_X, train_y):
    HandWrittenNumberFromZerotoNine = {}

    for x, y in zip(train_X, train_y):
        if y not in HandWrittenNumberFromZerotoNine:
            HandWrittenNumberFromZerotoNine[y] = []
            HandWrittenNumberFromZerotoNine[y].append(x)

    return HandWrittenNumberFromZerotoNine


def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2


def ReLU(z):
    return np.maximum(0, z)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def forward_prop(W1, B1, W2, B2, X):
    z1 = W1.dot(X) + B1
    a1 = ReLU(z1)
    z2 = W2.dot(a1) + B2
    a2 = softmax(z2)

    return z1, a1, z2, a2


def one_hot(y):
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    y_one_hot = y_one_hot.T
    return y_one_hot


def derivative_ReLU(z):
    return z > 0


def back_prop(z1, A1, z2, A2, W2, y, x):
    m = y.size
    one_hot_y = one_hot(y)

    dz2 = A2 - one_hot_y
    dW2 = 1 / m * dz2.dot(A1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)

    dz1 = W2.T.dot(dz2) * derivative_ReLU(z1)
    dW1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def updata_params(W1, B1, W2, B2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - dW1 * learning_rate
    B1 = B1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    B2 = B2 - db2 * learning_rate

    return W1, B1, W2, B2


def get_prediction(a2):
    return np.argmax(a2, axis=0)


def get_accuracy(a2, Y):
    return np.sum(get_prediction(a2) == Y) / Y.size


def gradient_descent(X, Y, epochs, learning_rate):
    W1, B1, W2, B2 = init_params()
    for i in range(epochs):
        z1, a1, z2, a2 = forward_prop(W1, B1, W2, B2, X)
        dW1, db1, dW2, db2 = back_prop(z1, a1, z2, a2, W2, Y, X)
        W1, B1, W2, B2 = updata_params(
            W1, B1, W2, B2, dW1, db1, dW2, db2, learning_rate)
        if i % 10 == 0:
            print(f"epoch: {i}")
            print(f"accuracy: {get_accuracy(a2, Y)}")

    return W1, B1, W2, B2


W1, B1, W2, B2 = gradient_descent(
    X_train.T, y_train, epochs=500, learning_rate=0.1)

data = np.ndarray(W1, W2, B1, B2)

np.save("HandwrittenNumbers.npy", data)
