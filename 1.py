import numpy as np
import nnfs
from nnfs.datasets import vertical_data


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        return prob


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = vertical_data(samples=100, classes=10)
dense1 = Layer_Dense(2, 32)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(32, 32)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(32, 10)
activation3 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense2_weights = dense2.weights.copy()
best_dense3_weights = dense3.weights.copy()
best_dense1_bias = dense1.biases.copy()
best_dense2_bias = dense2.biases.copy()
best_dense3_bias = dense3.biases.copy()

for iteration in range(200000):
    dense1.weights += 0.01 * np.random.randn(2, 32)
    dense2.weights += 0.01 * np.random.randn(32, 32)
    dense3.weights += 0.01 * np.random.randn(32, 10)
    dense1.biases += 0.01 * np.random.randn(1, 32)
    dense2.biases += 0.01 * np.random.randn(1, 32)
    dense3.biases += 0.01 * np.random.randn(1, 10)


    out = activation1.forward(dense1.forward(X))
    out = activation2.forward(dense2.forward(out))
    out = activation3.forward(dense3.forward(out))


    loss = loss_function.calculate(out, y)


    predictions = np.argmax(out, axis=1)
    accuracy = np.mean(predictions == y)


    if loss < lowest_loss:
        print("New set of weights found, iteration:",
              iteration, 'loss:', loss, 'acc :', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense3_weights = dense3.weights.copy()
        best_dense1_bias = dense1.biases.copy()
        best_dense2_bias = dense2.biases.copy()
        best_dense3_bias = dense3.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense2.weights = best_dense2_weights.copy()
        dense3.weights = best_dense3_weights.copy()
        dense1.biases = best_dense1_bias.copy()
        dense2.biases = best_dense2_bias.copy()
        dense3.biases = best_dense3_bias.copy()