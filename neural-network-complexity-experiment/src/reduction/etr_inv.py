"""
ETR-Inv Problem Module

Define ETR-Inv problem - the problem that Train-F2NN is reduced from

ETR-Inv is Existential Theory of Reals with Inversions:
∃X₁, ..., Xₙ ∈ ℝ : X ∈ [1/2, 2]ⁿ ∧ Φ(X)

Where Φ(X) is the conjunction of:
- Addition constraints: Xᵢ + Xⱼ = Xₖ
- Inversion constraints: Xᵢ · Xⱼ = 1
"""
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ConstraintType(Enum):
    """Type of constraints in ETR-Inv"""
    ADDITION = "addition"   # X + Y = Z
    INVERSION = "inversion"     # X * Y = 1


@dataclass
class Constraint:
    """Represent a constraint in ETR-Inv"""
    type: ConstraintType
    variables: List[str]

    def __post_init__(self):
        """Validate constraint"""
        if self.type == ConstraintType.ADDITION:
            if len(self.variables) != 3:
                raise ValueError("Addition constraint requires exactly 3 variables")

        elif self.type == ConstraintType.INVERSION:
            if len(self.variables) != 2:
                raise ValueError("Inversion constraint requires exactly 2 variables")

    def __repr__(self) -> str:
        if self.type == ConstraintType.ADDITION:
            return f"{self.variables[0]} + {self.variables[1]} = {self.variables[2]}"
        else:
            return f"{self.variables[0]} * {self.variables[1]} = 1"


class ETRInvProblem:
    """
    ETR-Inv Problem Instance

    Example from paper:
        ∃X₁, X₂, X₃, X₄ ∈ ℝ : (X₁ + X₂ = X₃) ∧ (X₁ + X₃ = X₄) ∧ (X₁ · X₄ = 1) ∧ (X₄ · X₃ = 1)
    """

    def __init__(self,
                 variables: List[str],
                 constraints: List[Tuple[str, ...]]):
        """
        Args:
            variables: List of variable names (e.g., ['X1', 'X2', 'X3', 'X4'])
            constraints: List of constraints as tuples:
                - ('add', 'X1', 'X2', 'X3') for X1 + X2 = X3
                - ('inv', 'X1', 'X2') for X1 * X2 = 1
        """
        self.variables = variables
        self.constraints = self._parse_constraints(constraints)
        self._validate()

    def _parse_constraints(self, constraints: List[Tuple[str, ...]]) -> List[Constraint]:
        """Parse constraint tuples to Constraint objects"""
        parsed = []

        for c in constraints:
            if c[0] == 'add':
                parsed.append(Constraint(
                    type=ConstraintType.ADDITION,
                    variables=list(c[1:])
                ))
            elif c[0] == 'inv':
                parsed.append(Constraint(
                    type=ConstraintType.INVERSION,
                    variables=list(c[1:])
                ))
            else:
                raise ValueError(f"Unknown constraint type: {c[0]}")

        return parsed

    def _validate(self):
        """Validate problem instance"""
        # Check all constraint variables exist
        for constraint in self.constraints:
            for var in constraint.variables:
                if var not in self.variables:
                    raise ValueError(f"Unknown variable in constraint: {var}")

    def is_satisfiable(self, assignment: Dict[str, float]) -> bool:
        """
        Check whether the assignment satisfies all constraints

        Args:
            assignment: Dictionary mapping variable names to values

        Returns:
            True if assignment satisfies all constraints
        """
        # Check domain constraints
        for var, value in assignment.items():
            if not (0.5 <= value <= 2.0):
                return False

        # Check constraints
        for constraint in self.constraints:
            if constraint.type == ConstraintType.ADDITION:
                X, Y, Z = [assignment[v] for v in constraint.variables]
                if not np.isclose(X + Y, Z, rtol=1e-6):
                    return False

            elif constraint.type == ConstraintType.INVERSION:
                X, Y = [assignment[v] for v in constraint.variables]
                if not np.isclose(X * Y, 1.0, rtol=1e-6):
                    return False

        return True

    def find_solution_naive(self, num_samples: int = 1000) -> Dict[str, float] | None:
        """
        Naive solver: Random sampling (for demonstration purposes)

        Args:
            num_samples: Number of random samples to try

        Returns:
            Solution dictionary if found, None otherwise
        """
        np.random.seed(42)

        for _ in range(num_samples):
            # Random assignment in [0.5, 2.0]
            assignment = {
                var: np.random.uniform(0.5, 2.0)
                for var in self.variables
            }

            if self.is_satisfiable(assignment):
                return assignment

        return None

    def get_addition_constraints(self) -> List[Constraint]:
        """Get all addition constraints"""
        return [c for c in self.constraints if c.type == ConstraintType.ADDITION]

    def get_inversion_constraints(self) -> List[Constraint]:
        """Get all inversion constraints"""
        return [c for c in self.constraints if c.type == ConstraintType.INVERSION]

    def count_gadgets_needed(self) -> Dict[str, int]:
        """
        Count the number of gadgets needed for reduction

        Returns:
            Dictionary with gadget counts
        """
        # Each variable need 1 canonical variable gadget
        num_variable_gadgets = len(self.variables)

        # Each addition constraint need 3 variable gadgets (copies)
        num_addition_gadgets = len(self.get_addition_constraints())

        # Each inversion constraint need 1 inversion gadget (copies)
        num_inversion_gadgets = len(self.get_inversion_constraints())

        # Lower bound gadgets (2 per variable + 2 per inversion)
        num_lower_bound_gadgets = 2 * num_variable_gadgets + 2 * num_inversion_gadgets

        return {
            'variable_gadgets': num_variable_gadgets + 3 * num_addition_gadgets,
            'inversion_gadgets': num_inversion_gadgets,
            'lower_bound_gadgets': num_lower_bound_gadgets,
            'total_gadgets': (num_variable_gadgets + 3 * num_addition_gadgets + num_inversion_gadgets + num_lower_bound_gadgets)
        }

    def __repr__(self) -> str:
        constraints_str = " ∧ ".join(str(c) for c in self.constraints)
        return f"ETRInvProblem(∃{', '.join(self.variables)} : {constraints_str})"

def create_example_problem() -> ETRInvProblem:
    """
    Create example problem from paper

    ∃X₁, X₂, X₃, X₄ ∈ [1/2, 2]⁴ : (X₁ + X₂ = X₃) ∧ (X₁ + X₃ = X₄) ∧ (X₁ · X₄ = 1) ∧ (X₄ · X₃ = 1)
    """
    variables = ['X1', 'X2', 'X3', 'X4']

    constraints = [
        ('add', 'X1', 'X2', 'X3'),  # X₁ + X₂ = X₃
        ('add', 'X1', 'X3', 'X4'),  # X₁ + X₃ = X₄
        ('inv', 'X1', 'X4'),        # X₁ · X₄ = 1
        ('inv', 'X4', 'X3'),        # X₄ · X₃ = 1
    ]

    return ETRInvProblem(variables, constraints)


def create_simple_problem() -> ETRInvProblem:
    """
    Create a simple problem to test

    ∃X, Y ∈ [1/2, 2]² : X · Y = 1
    """
    variables = ['X', 'Y']
    constraints = [('inv', 'X', 'Y')]

    return ETRInvProblem(variables, constraints)


def create_addition_only_problem() -> ETRInvProblem:
    """
    Create problem with only addition constraints

    ∃X, Y, Z ∈ [1/2, 2]³ : X + Y = Z
    """
    variables = ['X', 'Y', 'Z']
    constraints = [('add', 'X', 'Y', 'Z')]

    return ETRInvProblem(variables, constraints)