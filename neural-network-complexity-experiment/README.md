# Neural Network Training Complexity Experiment

Implementation of paper **[Training Fully Connected Neural Networks is ∃R-Complete](https://arxiv.org/pdf/2204.01368)** by Bertschinger et al.

## Introduction

This project implements and visualizes the main results from the paper, demonstrating that training 2-layer fully connected neural networks with ReLU activation is ∃R-complete - a more difficult problem than NP-complete.

## Purpose
1. **Understand the computational complexity** of neural network training
2. **Implement gadgets** from the paper: Variable Gadget, Inversion Gadget, Addition Gadget
3. **Visualize reduction** from ETR-Inv problem to Train-F2NN problem
4. **Practical experiments** with small instances

## Project Structure
```
src/
├── core/                 # Core neural network components
│   ├── network.py       # Neural network architecture
│   ├── trainer.py       # Training algorithms
│   └── loss.py          # Loss functions
├── gadgets/             # Gadget implementations từ paper
│   ├── variable.py      # Variable gadget
│   ├── inversion.py     # Inversion gadget
│   └── addition.py      # Addition constraint
├── reduction/           # ETR-Inv to Train-F2NN reduction
│   ├── etr_inv.py       # ETR-Inv problem definition
│   └── reducer.py       # Reduction algorithm
├── experiments/         # Experiment scripts
│   └── run_experiments.py
└── utils/               # Utilities
    ├── visualization.py # Plotting và visualization
    └── config.py        # Configuration management
```

## Installation
```bash
# CLone repository

```