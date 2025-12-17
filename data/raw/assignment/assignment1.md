# ELEC 576 / COMP 576 – Fall 2025  
## Assignment 1

**Due: Oct 7, 2025, 11:59 p.m. via Canvas**

---

## Submission Instructions
Submit your report as a **PDF** and your code as a ZIP file named:

```
netid-assignment1.zip
```

Upload everything to Canvas.

---

## GPU Resource
You may use:

- **AWS GPU instances** (AWS Educate credits + GitHub Student Pack credits)  
- **Google Colab** (recommended for convenience)

---

# 1. Backpropagation in a Simple Neural Network

You will implement backpropagation for a 3-layer neural network. Starter code is provided in  
`three_layer_neural_network.py`.

---

## a) Dataset — Make Moons

Uncomment the dataset generation section:

```python
# generate and visualize Make-Moons dataset
X, y = generate_data()
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
```

Run it and include the figure in your report.

---

## b) Activation Functions

Implement:

### 1. `actFun(self, z, type)`
Where `type ∈ {'Tanh', 'Sigmoid', 'ReLU'}`.

### 2. Derive derivatives of:
- Tanh  
- Sigmoid  
- ReLU  

### 3. Implement:
`diff_actFun(self, z, type)`  
Compute derivatives for all three activations.

---

## c) Build the 3-Layer Network

Network structure:

- Input: 2 nodes  
- Hidden layer: variable size  
- Output: 2 nodes (probabilities for 2 classes)

Equations:

```
z1 = W1x + b1
a1 = actFun(z1)
z2 = W2a1 + b2
a2 = ŷ = softmax(z2)
```

Loss function (cross entropy):

```
L = -(1/N) Σ_n Σ_i y_n,i log(ŷ_n,i)
```

### Implement:

1. `feedforward(self, X, actFun)`  
   Computes probabilities.

2. `calculate_loss(self, X, y)`  
   Computes cross-entropy loss.

---

## d) Backpropagation

### 1. Derive gradients:
- ∂L/∂W2  
- ∂L/∂b2  
- ∂L/∂W1  
- ∂L/∂b1  

### 2. Implement in code:
`backprop(self, X, y)`

---

## e) Training

Training code is already provided.

### 1. Train using activation functions:
- Tanh  
- Sigmoid  
- ReLU  

Include figures and describe differences.

Remove dataset visualization (as instructed).

### 2. Vary hidden layer size  
Train again (use Tanh). Describe the effect on accuracy and decision boundary.

---

## f) Build a Deeper Network (n-layer)

Write a new file `n_layer_neural_network.py`.

Your implementation must support:

- Arbitrary number of layers  
- Arbitrary layer sizes  

### Suggested structure (optional):

1. Create class: `DeepNeuralNetwork(NeuralNetwork)`  
2. Override:
   - feedforward  
   - backprop  
   - calculate_loss  
   - fit_model  
3. Create a `Layer()` class  
4. Use it to build feedforward/backprop modularly  
5. Include L2 regularization in:
   - Loss  
   - Gradients  

### Experiments:

- Vary:
  - Number of layers  
  - Hidden sizes  
  - Activation functions  
  - Regularization  

Include:
- Decision boundary plots  
- Interesting observations  

### Train on a second dataset  
Pick any Scikit-learn dataset or another dataset you like.  
Describe:
- Dataset  
- Network configuration  
- Observations  

---

# 2. Training a Simple Deep Convolutional Network on MNIST

Starter code provided on Canvas.  
Review tutorial: **Getting Started with PyTorch**.

MNIST:
- 55,000 training  
- 10,000 test  
- 5,000 validation  
- Each image is 28×28  

---

## a) Build and Train a 4-Layer DCN

Architecture:

```
conv1(5×5×1→32) → ReLU → maxpool(2×2)
conv2(5×5×32→64) → ReLU → maxpool(2×2)
fc(1024) → ReLU → Dropout(0.5) → Softmax(10)
```

Steps:

1. Read ConvNet tutorial  
2. Load MNIST (use torchvision)  
3. Complete class `Net()`  
4. Complete training function `train()`  
5. Use TensorBoard to visualize training loss  
6. Report test accuracy  

Include TensorBoard plots.

---

## b) More Training Visualization

Add TensorBoard logging for each 100 iterations:

- Weights  
- Biases  
- Net inputs  
- ReLU activations  
- Max-pool activations  

Also log after each epoch:

- Validation error  
- Test error  

Include figures.

---

## c) More Experiments

Try different:

- Activations:
  - tanh, sigmoid, leaky-ReLU, MaxOut  
- Initialization:
  - Xavier  
- Training algorithms:
  - SGD  
  - Momentum  
  - Adagrad  

Include TensorBoard figures and descriptions.

---

# Collaboration Policy

Collaboration allowed for ideas.  
Write-ups must be individual.

---

# Plagiarism Policy

No plagiarism allowed.  
Cite all sources.

---

# LLM Policy

**LLMs (ChatGPT, Copilot, etc.) are *not permitted* for coding in Assignment 1.**  
All coding must be your own.

