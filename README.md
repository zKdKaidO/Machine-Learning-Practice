# Machine Learning from Scratch: Theory & NumPy Implementation

Welcome to my journey of building Machine Learning algorithms from the ground up!

This repository is dedicated to demystifying the "black box" of Machine Learning. Instead of relying on high-level frameworks like scikit-learn or PyTorch, I am implementing core algorithms using pure Python and **NumPy**. 

The goal is to deeply understand the mathematics, architecture, and inner workings of these models by bridging the gap between theoretical concepts and actual code.

## 🎯 Project Goals

* **Deep Understanding:** To truly comprehend how algorithms work, not just how to call them.
* **Theory meets Code:** Every implementation is paired with the mathematical theory that drives it (e.g., Matrix Calculus, Probability, Optimization).
* **Minimal Dependencies:** Relying primarily on `numpy` for matrix operations and `matplotlib` (optional) for visualization. No `sklearn`, `tensorflow`, or `pytorch` for the core logic.
* **Clean Architecture:** Designing the code using Object-Oriented Programming (OOP) principles (like modular Layers, Loss functions, and Optimizers) to mimic the structure of professional frameworks.

## 📂 Implemented Models

Here is the current list of algorithms I have successfully implemented from scratch:

### Probabilistic Models
* **Naive Bayes Classifier:** Implementing Bayes' Theorem for classification tasks.
* **Hidden Markov Models (HMM):** Modeling sequential data using states, transition probabilities, and emission probabilities.

### Neural Networks (Deep Learning Basics)
* **Multilayer Perceptron (MLP):** Building a fully functional feedforward neural network.
    * *Features:* Custom Fully Connected (Linear) Layers, Activation Functions (ReLU, Sigmoid), and Backpropagation using the Chain Rule.
    * *Loss Functions:* MSE, MAE, Huber Loss.
    * *Optimizers:* Stochastic Gradient Descent (SGD), AdamW (with Momentum, RMSprop, and Decoupled Weight Decay).

## 🚀 Upcoming Implementations (Roadmap)

I am constantly adding new models to this repository. Here is what I plan to build next:

### Regression & Linear Models
* [ ] Linear Regression (using Gradient Descent and Normal Equation)
* [ ] Logistic Regression (for binary classification)

### Tree-Based Models
* [ ] Decision Trees (implementing Entropy/Gini Impurity for splitting)
* [ ] Random Forests (Ensemble learning)

### Unsupervised Learning
* [ ] K-Means Clustering
* [ ] Principal Component Analysis (PCA)

## 💡 How to Use This Repository

Each algorithm is contained within its own folder, typically including:
1.  **`model.py`:** The core NumPy implementation (the classes and math).
2.  **`train.py` / `example.py`:** A script demonstrating how to instantiate the model, train it on sample data, and evaluate it.
3.  **`theory.md` (Optional):** Mathematical proofs or explanations (like derivation of gradients for Backpropagation).

Feel free to explore the code, run the examples, and use them as learning materials!

## 🤝 Contributing & Feedback

This is primarily a personal learning project, but I welcome any feedback, discussions on mathematical optimizations, or suggestions for making the NumPy code more efficient. If you spot a bug or have an idea, feel free to open an issue!

---
*“What I cannot create, I do not understand.” - Richard Feynman*