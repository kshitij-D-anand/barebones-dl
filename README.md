# barebones-dl

A zero-dependency deep learning engine built in pure NumPy.

A hands-on implementation of neural network fundamentals (inspired by 'neural network from scratch').
It implements core deep learning building blocks, from scratch matrix multiplications and automatic gradient backpropagation to custom optimizers and regularizers.

## 🛠️ Features

- **Layers:** Fully Connected (Dense), Dropout
- **Activations:** ReLU, Softmax
- **Loss Functions:** Categorical Cross-Entropy, Mean Squared Error (MSE)
- **Optimizers:** Linear GD, SGD (with Momentum), AdaGrad, RMSprop, Adam
- **Regularization:** L1 & L2 Weight/Bias Regularization, Forward Dropout
- **Benchmarks:** Classification (Spiral Dataset) & Regression (Sine Wave)

## Repository Structure

```text
barebones-dl/
├── barebones_dl/          # Core framework library
│   ├── __init__.py        # Exposes clean top-level API imports
│   ├── layers.py          # Dense & Dropout layers
│   ├── activations.py     # ReLU & Softmax activations
│   ├── losses.py           # Cross-Entropy & MSE losses
│   └── optimizers.py      # SGD, RMSprop & Adam optimizers
├── spiral_train.py        # Classification benchmark script
├── train_sine.py          # Regression benchmark script
├── requirements.txt       # Project dependencies (NumPy, nnfs, Matplotlib)
└── README.md
```
