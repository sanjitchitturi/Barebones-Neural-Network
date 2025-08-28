# Barebones Neural Network

A **minimal two-layer neural network** implemented **from scratch in NumPy** for binary classification.  
Trains on synthetic data (`make_blobs`) and visualizes the **decision boundary**.  

No TensorFlow. No PyTorch. Just pure NumPy + a bit of math.  

---

## Features
- Full forward + backward pass implemented manually.
- ReLU (hidden layer) and Sigmoid (output) activations.
- He initialization for better training stability.
- Accuracy reporting on **train & test**.
- Simple decision boundary visualization.
- No frameworks — just **NumPy** and **matplotlib**.

---

## Installation

Clone the repo:
```bash
git clone https://github.com/sanjitchitturi/Barebones-Neural-Network.git
cd Barebones-Neural-Network
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the training script:

```bash
python main.py
```

You’ll see logs like:

```
Epoch 000 | Loss: 0.6931
Epoch 100 | Loss: 0.2785
Epoch 200 | Loss: 0.1452
...
Final Results:
Train Accuracy: 1.00
Test Accuracy:  0.95
```

A decision boundary plot will also be saved as:

```
decision_boundary.png
```

---

## Requirements

Listed in `requirements.txt`:

```
numpy
matplotlib
scikit-learn
```

---

## How it Works

1. Generate 2-class synthetic data with `make_blobs`.
2. Initialize weights with **He initialization**.
3. Forward pass:

   * Hidden layer: `ReLU`
   * Output layer: `Sigmoid`
4. Backward pass:

   * Binary cross-entropy loss
   * Gradient descent updates
5. Plot decision boundary + accuracy results.

---
