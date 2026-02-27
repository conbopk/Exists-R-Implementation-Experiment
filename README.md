# Neural Network Training Complexity

## Paper Implementation
This project is a complete implementation of the paper **[Training Fully Connected Neural Networks is ∃R-Complete](https://arxiv.org/pdf/2204.01368)** by Bertschinger et al. (NeurIPS 2023).

## Main Result
Paper proves that **training 2-layer fully connected neural networks with ReLU activation is ∃R-complete** - a problem harder than NP-complete.

### Key Insights
1. **∃R Complexity Class**: Class of problems polynomial-time equivalent to finding real roots of multivariate polynomials
2. **Reduction**: ETR-Inv (Existential Theory of Reals with Inversions) ≤_p Train-F2NN
3. **Implication**: Training neural networks is fundamentally hard - there cannot be a combinatorial search algorithm (unless NP = ∃R)

## Implementation Architecture
### Project Structure
```
neural-network-complexity-experiment/
├── src/
│   ├── core/               # Core components
│   │   ├── network.py     # 2-layer ReLU network
│   │   ├── loss.py        # MSE loss, dataset handling
│   │   └── trainer.py     # Training algorithms (SGD, Adam)
│   │
│   ├── gadgets/           # Gadget implementations từ paper
│   │   ├── variable.py    # Variable gadget (encode X ∈ [0.5, 2])
│   │   └── inversion.py   # Inversion gadget (encode X·Y = 1)
│   │
│   ├── reduction/         # Reduction from ETR-Inv to Train-F2NN
│   │   ├── etr_inv.py     # ETR-Inv problem definition
│   │   └── reducer.py     # Reduction algorithm
│   │
│   ├── experiments/       # Experiment scripts
│   │   └── run_experiments.py
│   │
│   └── utils/             # Utilities
│       ├── config.py      # Configuration management
│       └── visualization.py # Plotting tools
│
├── tests/                 # Unit tests
│   └── test_gadgets.py
│
├── results/               # Experiment results
│   ├── plots/            # Visualizations
│   ├── logs/             # Training logs
│   └── metrics/          # Performance metrics
│
├── data/                  # Data directory
├── README.md              # Main documentation
├── setup.py               # Package installation
├── requirements.txt       # Dependencies
└── config.yaml            # Configuration file
```

## Key Components Explained

### 1. Variable Gadget (`src/gadgets/variable.py`)

**Purpose**: Encode a real variable X ∈ [0.5, 2] using the slope of a piecewise linear function.

**How it works**:
- Gadget has 9 parallel data lines
- Slope s_X = X + 1 encodes value X
- Function must fit exactly all data points
- There are 4 breaklines parallel with data lines

**Key method**:
```python
gadget = VariableGadget(value=1.5)  # X = 1.5, slope = 2.5
dataset = gadget.generate_data_points()
```

### 2. Inversion Gadget (`src/gadgets/inversion.py`)

**Purpose**: Encode constraint nonlinear X·Y = 1

**How it works**:
- Use 2 output dimensions
- Dimension 1 encodes X, dimension 2 encodes Y
- 13 data lines with different labels at 2 dimensions
- 5 breaklines enforce nonlinear relationship

**Key insight**: Geometry of breaklines force s_X · s_Y = constant, from which X·Y = 1

### 3. Reduction Pipeline (`src/reduction/reducer.py`)

**Purpose**: Reduce ETR-Inv problem to Train-F2NN problem

**Steps**:
1. Parse ETR-Inv instance (variables + constraints)
2. Create variable gadgets for each variable
3. Create inversion gadgets for each inversion constraint
4. Add constraint points to encode additions and copies
5. Combine into a dataset
6. Target error γ = 0 (exact fit required)

**Complexity**:
- Number of data points: O(n + m) with n variables, m constraints
- Minimum hidden neurons: 4n + 5k with k inversion constraints

### 4. Neural Network (`src/core/network.py`)

**Architecture**: 2-layer fully connected with ReLU

```
Input (2D) → Hidden (ReLU) → Output (2D)
```

**Key features**:
- Piecewise linear output
- Breaklines at ReLU activation boundaries
- Exact gradient computation for training

### 5. Training (`src/core/trainer.py`)

**Algorithm**:
- Vanilla SGD
- Adam optimizer (recommended)

**Features**:
- Early stopping based on tolerance
- Training history tracking
- Multi-start training to overcome non-convexity

## Experiments

### Experiment 1: Variable Gadget Test
Verify variable gadget operates correctly and network can fit exactly.

### Experiment 2: Inversion Gadget Test
Test nonlinear constraint X·Y = 1 with 2 output dimensions.

### Experiment 3: Addition Constraint Test
Test linear constraint X + Y = Z by measuring line intersections.

### Experiment 4: Full Reduction Test
Complete reduction from ETR-Inv instance to Train-F2NN problem.

## Visualization Features

### Gadget Visualizations
- Top-down view of data lines
- Cross-section view showing piecewise linear function
- Measuring lines and expected contributions

### Training Visualizations
- Loss curves (MSE and max error)
- Convergence plots
- Prediction vs target scatter plots
- Breakline geometry

### Example Output
```
Training Loss: 1.23e-07
Max Error: 3.45e-04
Converged: True
Time: 45.2s
```

## Getting Started

### Quick Setup
```bash
git clone https://github.com/conbopk/Exists-R-Implementation-Experiment.git
cd neural-network-complexity-experiment
pip install -r requirements.txt
python -m src.experiments.run_experiments
```

### Run Single Experiment
```python
from src.experiments.run_experiments import experiment_1_variable_gadget
gadget, network, history = experiment_1_variable_gadget()
```