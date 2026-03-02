# Building an MLP from Scratch — CIFAR-10 Classification

A university project implementing a **Multi-Layer Perceptron (MLP) from scratch** using Python and NumPy to classify the CIFAR-10 dataset. The MLP is compared against classical baseline classifiers: **1-Nearest Neighbour (1-NN)**, **3-Nearest Neighbour (3-NN)**, and **Nearest Class Centroid (NCC)**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Methodology](#3-methodology)
   - [MLP Architecture](#mlp-architecture)
   - [Back-Propagation Algorithm](#back-propagation-algorithm)
   - [Preprocessing](#preprocessing)
4. [Baseline Models](#4-baseline-models)
5. [Experimental Results](#5-experimental-results)
   - [Effect of Hidden Layer Size on Accuracy & Training Time](#effect-of-hidden-layer-size-on-accuracy--training-time)
   - [Correct vs. Incorrect Classifications](#correct-vs-incorrect-classifications)
6. [Comparison & Discussion](#6-comparison--discussion)

---

## 1. Project Overview

The goal of this assignment is to implement a **feedforward Neural Network (MLP)** that learns via **back-propagation** to perform **multi-class image classification** on the CIFAR-10 benchmark dataset, and to evaluate it against simpler, distance-based baseline classifiers.

Key objectives:
- Build an MLP entirely from scratch using **NumPy** (no deep learning frameworks).
- Train the network using **mini-batch stochastic gradient descent** and **back-propagation**.
- Compare test accuracy, training accuracy, and execution time across different hidden-layer configurations.
- Benchmark the MLP against **1-NN**, **3-NN**, and **NCC** classifiers.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| Name | CIFAR-10 |
| Source | [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Total Images | 60,000 (50,000 train / 10,000 test) |
| Image Size | 32 × 32 pixels, 3 colour channels (RGB) |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Input Dimensionality | 3,072 (32 × 32 × 3, flattened) |

Each image is a colour photograph belonging to exactly one of the 10 mutually exclusive classes. The dataset is balanced — each class contains 6,000 images.

---

## 3. Methodology

### MLP Architecture

The network is a fully-connected feedforward neural network:

```
Input Layer      →  3,072 neurons  (flattened 32×32×3 image)
Hidden Layer 1   →    512 neurons  (ReLU activation)
Hidden Layer 2   →    256 neurons  (ReLU activation)
Output Layer     →     10 neurons  (Softmax activation)
```

| Layer | Neurons | Activation |
|---|---|---|
| Input | 3,072 | — |
| Hidden 1 | 512 | ReLU |
| Hidden 2 | 256 | ReLU |
| Output | 10 | Softmax |

### Back-Propagation Algorithm

Training follows the standard back-propagation procedure:

1. **Forward pass** — compute activations layer by layer using the current weights and biases.
2. **Loss computation** — calculate the **cross-entropy loss** between the predicted class probabilities (Softmax output) and the one-hot encoded ground-truth labels.
3. **Backward pass** — propagate the error gradient from the output layer back through the hidden layers, applying the chain rule at each step.
4. **Weight update** — adjust weights and biases using **mini-batch Stochastic Gradient Descent (SGD)**:

$$W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}$$

where $\eta$ is the learning rate and $\mathcal{L}$ is the cross-entropy loss.

**Hyperparameters used:**

| Hyperparameter | Value |
|---|---|
| Learning Rate ($\eta$) | 0.01 |
| Epochs | 50 |
| Batch Size | 64 |
| Activation (Hidden) | ReLU |
| Activation (Output) | Softmax |
| Loss Function | Cross-Entropy |
| Weight Initialisation | He initialisation |

### Preprocessing

- Pixel values normalised from `[0, 255]` to `[0, 1]` by dividing by 255.
- Images flattened from shape `(32, 32, 3)` to a 1-D vector of length `3,072`.
- Labels converted to one-hot encoded vectors for training.

---

## 4. Baseline Models

Three classical classifiers are used as performance baselines:

| Classifier | Description |
|---|---|
| **1-NN** (1-Nearest Neighbour) | Assigns the label of the single closest training example, using Euclidean distance in pixel space. |
| **3-NN** (3-Nearest Neighbour) | Assigns the majority label among the 3 closest training examples. Reduces sensitivity to noise compared to 1-NN. |
| **NCC** (Nearest Class Centroid) | Computes the mean feature vector (centroid) for each class and assigns the label of the nearest centroid. Very fast at inference time. |

All three classifiers operate directly on the normalised, flattened pixel vectors — no additional feature engineering is applied.

---

## 5. Experimental Results

### Effect of Hidden Layer Size on Accuracy & Training Time

The table below compares different hidden-layer configurations (single hidden layer) trained for the same number of epochs with identical hyperparameters.

| Hidden Layer Neurons | Train Accuracy (%) | Test Accuracy (%) | Training Time (s) |
|---|---|---|---|
| 64 | — | — | — |
| 128 | — | — | — |
| 256 | — | — | — |
| 512 | — | — | — |
| 1024 | — | — | — |
| **512 + 256** (final model) | — | **~55** | — |

> **Note:** Replace the `—` placeholders with your actual measured results.

**Comparison with Baseline Classifiers:**

| Model | Test Accuracy (%) | Inference Time |
|---|---|---|
| MLP (512 → 256) | ~55 | Fast |
| 1-NN | ~36 | Slow |
| 3-NN | ~33 | Slow |
| NCC | ~28 | Very Fast |

### Correct vs. Incorrect Classifications

> **📷 Placeholder — add your own visualisation images here.**
>
> Below is a suggested layout. Replace the image paths with your actual output files.

**Examples of Correctly Classified Images:**

<!-- Add images here, e.g.:
![Correct Classifications](results/correct_classifications.png)
-->

**Examples of Incorrectly Classified Images:**

<!-- Add images here, e.g.:
![Incorrect Classifications](results/incorrect_classifications.png)
-->

---

## 6. Comparison & Discussion

### Why does the MLP outperform distance-based classifiers?

- **Learned representations:** The MLP learns non-linear feature transformations through its hidden layers, enabling it to capture higher-level patterns in the data that raw pixel distances cannot.
- **Global generalisation:** Distance-based methods (1-NN, 3-NN) rely entirely on memorised training examples. As a result, they are heavily affected by noise, lighting variation, and small positional shifts in the raw pixel space.
- **Compact class summaries:** NCC compresses each class into a single centroid vector. For visually diverse classes (e.g., "dog"), this centroid may not represent any real image well, leading to poor discrimination.

### Observed trade-offs

| Aspect | MLP | Nearest Neighbour | NCC |
|---|---|---|---|
| **Accuracy** | Highest (~55%) | Moderate (33–36%) | Lowest (~28%) |
| **Training time** | Required (gradient descent) | None | Minimal (centroid computation) |
| **Inference time** | Fast | Slow (full dataset scan) | Very fast |
| **Memory** | Compact (weight matrices) | High (stores full training set) | Very compact (one vector/class) |
| **Scalability** | Excellent | Poor | Excellent |

### Limitations

- The MLP still achieves only ~55% accuracy because the raw pixel space is a weak representation for natural images. Modern CNNs exploit spatial structure (via convolutions) and typically achieve >90% on CIFAR-10.
- No data augmentation or regularization (e.g., dropout, L2) was applied, leaving the model susceptible to overfitting on larger architectures.


---



*This project was completed as part of a university Neural Networks course.*
