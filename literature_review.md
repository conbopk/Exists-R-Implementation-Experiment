### **LITERATURE REVIEW: COMPLEXITY ANALYSIS OF NEURAL NETWORK TRAINING**

1.1. **Background and Motivation**

Neural Network Training is one of the fundamental problems in machine learning. From the perspective of computational complexity theory, we need to accurately classify the difficulty of this problem.

**Historical Development:**
- **Before 2021:** Results mainly proved NP-hardness for specific neural network architectures
- **NeurIPS 2021:** Abrahamsen, Kleist, and Miltzow proved ∃R-completeness for fully connected 2-layer networks with linear activation functions, but the network architecture was "adversarial" (intentionally difficult)
- **NeurIPS 2023:** Bertschinger et al. (this paper) strengthened the result to ∃R-completeness for **fully connected** 2-layer ReLU networks—the most practical and common architecture

1.2. **Main Contributions of the Paper**

| Aspect           | Previous Results            | New Results (This Paper)   |
| ---------------- | --------------------------- | -------------------------- |
| **Architecture** | Adversarial (intentionally hard) | Fully connected (common)   |
| **Activation**   | Linear                      | ReLU (most common)         |
| **Input/Output** | Unrestricted                | 2 inputs, 2 outputs        |
| **Hardness**     | ∃R-complete                 | ∃R-complete (tighter)      |

**Theorem 3 (Main Result):** Train-F2NN (Training Fully Connected 2-layer Neural Network) is ∃R-complete, even when:
- There are only 2 input neurons
- There are only 2 output neurons
- The number of data points is linear in the number of hidden neurons
- Only 13 different labels are used
- The target error is γ = 0
- The ReLU activation function is used

1.3. **Complexity Class ∃R (Existential Theory of the Reals)**

**Definition:** ∃R is the class of decision problems polynomial-time equivalent to finding real roots of systems of multivariate polynomial equations.

**Relationships:**
```
NP ⊆ ∃R ⊆ PSPACE
```

**Practical Implications:**
- If ∃R ≠ NP (widely believed), then training neural networks is **harder** than NP-complete problems
- No efficient combinatorial algorithm exists for the multi-output case (unlike Arora et al. for single output)
- Explains why global optimization algorithms for multi-output networks do not exist

1.4. **Proof Techniques**

**Reduction from ETR-Inv:**
- ETR-Inv: A variant of ETR with only two types of constraints: X + Y = Z and X · Y = 1
- Proven ∃R-complete by Abrahamsen et al.

**Main Gadgets:**

1. **Variable Gadget:** Encodes a real variable X ∈ [1/2, 2] as the slope s_X = X + 1 of a piecewise linear segment
    - 12 parallel data lines
    - 4 mandatory breaklines
    - Weak data point to bound the slope

2. **Addition Gadget:** Encodes X + Y = Z
    - Uses intersection of measuring lines
    - Data point at intersection with label 10

3. **Inversion Gadget:** Encodes X · Y = 1 (nonlinear constraint)
    - Superposition of 2 variable gadgets
    - 13 data lines, 5 breaklines
    - Two output dimensions with nonlinear dependency: s_X · s_Y = s_X + s_Y

4. **Lower Bound Gadget:** Replaces weak data points with ordinary data points

**Key Insight:** Adding a second output dimension enables encoding of nonlinear constraints (inversion), increasing complexity from NP to ∃R.

1.5. **Algebraic Universality (Theorem 4)**

A stronger result than ∃R-completeness: For every algebraic number α, there exists an instance that has a solution in ℚ[α] but no solution in any field not containing α.

**Consequences:**
- Infinite precision (arbitrary precision) is required to represent optimal weights
- Numerical methods cannot find exact solutions
- Symbolic computation is mandatory

1.6. **Comparison with Related Work**

| Paper                         | Architecture                | Activation | Result          | Notes                    |
| ----------------------------- | --------------------------- | ---------- | --------------- | ------------------------ |
| Arora et al. [6]              | 2-layer, 1 output           | ReLU       | ∈ NP            | Combinatorial algorithm  |
| Abrahamsen et al. [3]         | 2-layer (adversarial)       | Linear     | ∃R-complete     | Unnatural architecture   |
| Froese & Hertrich [32]        | 2-layer, 2 inputs, 1 output | ReLU       | NP-hard         | Fixed dimension          |
| **Bertschinger et al. (this)**| **2-layer fully connected** | **ReLU**   | **∃R-complete** | **Most practical**       |

1.7. **Limitations and Open Problems**

**Limitations:**
- Proof is for γ = 0 (exact fitting)
- ReLU is piecewise linear (does not apply to Sigmoid, soft ReLU)
- 2D input (can be easily extended)

**Open Problems:**
- Complexity with 1D input and multi-output?
- Is approximate training (γ > 0) in NP?
- Deep networks (3+ layers) and CNNs?
