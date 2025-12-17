# ELEC 576 / COMP 576 – Fall 2025  
## Assignment 2

**Due: November 6, 2025, 11:59 PM via Canvas**

---

## Submission Instructions

Submit your report as a **PDF** on Canvas.

You may choose:

### **Option 1**
- Submit all answers, screenshots, and figures in a single PDF report.
- Submit all code and supporting files as a ZIP named:
```
netid-assignment2.zip
```

### **Option 2**
- Submit a **PDF export of your Jupyter Notebook** (must include all code + outputs).
- Make sure to run all cells before exporting.

Temporary files (e.g., TensorBoard logs) should NOT be included.

---

## GPU Resource

You may use:

- **AWS GPU instances** (AWS Educate credits + GitHub Student Pack)
- **Google Colab**

---

# 1. Visualizing a CNN with CIFAR10

Train a CNN on CIFAR10 and visualize early-layer filters and activations.

---

## a) CIFAR10 Dataset

CIFAR10 consists of **32×32 color images** in 10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Two import options:

### **Option A (Recommended): TorchVision**
- Images are RGB  
- Resolution: 32×32  

### **Option B**
- Use `trainCifarStarterCode.py` and provided ZIP  
- Images are **grayscale 28×28**  
- Requires preprocessing + one-hot labels  

---

## b) Train LeNet5 on CIFAR10

Implement and train the following variant of **LeNet5**:

- Conv (5×5), 6 filters → tanh  
- MaxPool (2×2)  
- Conv (5×5), 16 filters → tanh  
- MaxPool (2×2)  
- FC: 5×5×16 → 120 → 84 → 10  
- Softmax output  

### Required outputs:
- Training accuracy plot  
- Testing accuracy plot  
- Training loss plot  
- Hyperparameter experiments (LR, momentum, optimizer, etc.)

---

## c) Visualize the Trained Network

### 1. Visualize first-layer convolution filters  
They should resemble **Gabor filters** (edge detectors).

### 2. Visualize activations on test images  
Include summary statistics for each convolutional layer.

---

# 2. Visualizing and Understanding Convolutional Networks

Read the paper:  
**"Visualizing and Understanding Convolutional Networks" — Zeiler & Fergus**

### Task:
- Summarize the key ideas of the paper.

### Optional:
Apply a visualization method (e.g., deconvolutional network) to the model trained in Problem 1.

---

# 3. Build and Train an RNN on MNIST

Use the starter code `rnnMNISTStarterCode.py`.

MNIST images are 28×28; the RNN will process input **one row (28 pixels) at a time**.

---

## a) Set Up an RNN

Modify the following:

- Hidden layer size  
- Learning rate  
- Training iterations  
- Cost function (use softmax cross entropy with logits)  
- Optimizer  

---

## b) Try LSTM or GRU

Experiment using:

- `torch.nn.GRU`
- `torch.nn.LSTM`

### Required outputs:
- Train accuracy  
- Test accuracy  
- Train loss  

Try varying hidden units and compare performance.

---

## c) Compare Against the CNN

Compare results from:

- This RNN  
- The CNN you built in Assignment 1  

Discuss differences in:

- Accuracy  
- Training behavior  
- Strengths/weaknesses  

---

# Collaboration Policy

Collaboration is encouraged for discussing ideas, but all write-ups must be done **independently**.

---

# Plagiarism Policy

Plagiarism is strictly prohibited.  
Cite all external sources explicitly.

