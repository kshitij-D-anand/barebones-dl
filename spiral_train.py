import numpy as np
from nnfs.datasets import spiral_data
import nnfs

# Import your custom engine components
from barebones_dl import (
    Layer_Dense,
    Layer_Dropout,
    Activation_ReLU,
    Activation_Softmax_Loss_CategoricalCrossEntropy,
    Optimizer_ADAM,
)

nnfs.init()
coords, labels = spiral_data(samples=1000, classes=3)

# Initialisatin
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

optimizer = Optimizer_ADAM(learning_rate=0.05, decay=5e-5)

for iteration in range(10001):
    # Forward Pass
    dense1.forward(coords)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    
    data_loss = loss_activation.forward(dense2.output, labels)

    regularization_loss = (
        loss_activation.loss.regularization_loss(dense1) + 
        loss_activation.loss.regularization_loss(dense2)
    )
    loss = data_loss + regularization_loss

    # Calculate Accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(labels.shape) == 2:
        labels = np.argmax(labels, axis=1)
    accuracy = np.mean(predictions == labels)

    if not iteration % 1000:
        print(f"Iteration: {iteration:<5} | Accuracy: {accuracy:.3f} | Loss: {loss:.3f}")

    # Backward Pass
    loss_activation.backward(loss_activation.output, labels)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # Update Weights and Biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# tESTING
print("\n--- Running Validation ---")
test_coords, test_labels = spiral_data(samples=100, classes=3)

# Forward pass without dropout for testing
dense1.forward(test_coords)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, test_labels)

predictions = np.argmax(loss_activation.output, axis=1)
if len(test_labels.shape) == 2:
    test_labels = np.argmax(test_labels, axis=1)
accuracy = np.mean(predictions == test_labels)

print(f"Validation Performance | Accuracy: {accuracy:.3f} | Loss: {loss:.3f}")