import numpy as np
from .activations import Activation_Softmax

class Loss:
    def regularization_loss(self, layer):
        regularization_loss = 0
        # L1 regularization - weights
        # Calculate only when factor > 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        
        # L1 regularization - biases
        # Calculate only when factor > 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def calculate(self, output, y, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clipping the value between (0, 1) such that values are never 0 neither 1
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Checking shape for two cases, either array of given indices {e.g. [1, 0, 1]} or One-Hot-Vectors [[0, 1, 0], [1, 0, 0], [0, 1, 0]] 
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate Gradient
        self.dinputs = -y_true / dvalues
        # Normalise Gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        # Output Layer activation fucntion
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Calculate and return the Loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # No. of samples
        samples = len(dvalues)

        # If values are One-Hot Encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copying to safely modify
        self.dinputs = dvalues.copy()
        # Calculating Gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalising Gradient
        self.dinputs = self.dinputs / samples