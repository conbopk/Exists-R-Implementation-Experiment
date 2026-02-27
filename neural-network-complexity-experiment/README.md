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
# Clone repository
git clone https://github.com/conbopk/Exists-R-Implementation-Experiment.git
cd neural-network-complexity-experiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run a simple experiment

```python
from src.experiments.run_experiments import experiment_1_variable_gadget
gadget, network, history = experiment_1_variable_gadget()
```

### 2. Visualize gadgets

```python
from src.gadgets import VariableGadget
from src.utils import visualize

# Create and visualize a variable gadget
gadget = VariableGadget(value=1.5)
visualize.plot_gadget(gadget)
```

### 3. Test reduction

```python
from src.reduction import ETRInvProblem, Reducer

# Define ETR-Inv instance
problem = ETRInvProblem(
    variables=['X1', 'X2', 'X3', 'X4'],
    constraints=[
        ('add', 'X1', 'X2', 'X3'),  # X1 + X2 = X3
        ('inv', 'X1', 'X4')          # X1 * X4 = 1
    ]
)

# Reduce to Train-F2NN
reducer = Reducer()
nn_problem = reducer.reduce(problem)
```

## Experiments

Project includes the following experiments:
1. **Variable Gadget Test**: Verify that variable gadget works properly
2. **Inversion Gadget Test**: Test nonlinear constraint XY = 1
3. **Addition Constraint Test**: Test linear constraint X + Y = Z
4. **Full Reduction Test**: Complete reduction test from ETR-Inv

Run all experiments:
```bash
python -m src.experiments.run_experiments
```

## Result

The results of the experiments will be saved in `results/`:
- `plots/`: Visualizations
- `logs/`: Training logs
- `metrics/`: Performance metrics

## Key Concepts from Paper

### 1. Variable Gadget
Encode a real variable X ∈ [1/2, 2] using the slope of a piecewise linear function.

### 2. Inversion Gadget
Encode the nonlinear constraint XY = 1 using 2 output dimensions.

### 3. Addition Constraint
Encode linear constraint X + Y = Z equal to data points at measuring lines.

### 4. ∃R-Completeness
Prove that Train-F2NN has complexity equivalent to finding real polynomial roots.

## References

- Bertschinger et al. "Training Fully Connected Neural Networks is ∃R-Complete" (NeurIPS 2023)
- ETR-Inv problem definition
- ∃R complexity class

## Contributing

Contributions are welcome! Please:
1. Fork the repo
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License