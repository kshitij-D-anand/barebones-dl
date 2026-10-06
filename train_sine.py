import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from nnfs.datasets import sine_data
import nnfs

from barebones_dl import (
    Layer_Dense,
    Activation_ReLU,
    Optimizer_ADAM
)

nnfs.init()

X, y = sine_data()

# Initialization
dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
optimizer = Optimizer_ADAM(learning_rate=0.005, decay=1e-3)

for epoch in range(10001):
    # forward
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)

    data_loss = np.mean((y - dense3.output) ** 2)

    if not epoch % 1000:
        print(f"Epoch:{epoch} \tMSE Loss:{data_loss:.5f}")

    # backward
    dinputs = -2 * (y - dense3.output) / len(y)
    dense3.backward(dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and baises
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

print("\n--- Plotting Model Predictions vs Ground Truth ---")
plt.plot(X, y, label="Target (Sine Wave)", color="blue")
plt.plot(X, dense3.output, label="BareBonesDL Fit", color="red", linestyle="--")
plt.title("BareBonesDL - Regression Benchmark")
plt.legend()
plt.show()