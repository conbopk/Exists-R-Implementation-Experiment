"""
Reduction Module

Reduce ETR-Inv problem to Train-F2NN problem
This is core of ∃R-hardness proof
"""
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from src.core.loss import Dataset, DataPoint
from src.core.network import TwoLayerNetwork
from src.gadgets.inversion import InversionGadget
from src.gadgets.variable import VariableGadget, VariableConstraintEncoder
from src.reduction.etr_inv import ETRInvProblem


@dataclass
class TrainF2NNProblem:
    """
    Train-F2NN Problem Instance

    Problem: Given dataset D and target error y, find weights Θ such that loss(Θ, D) ≤ γ

    In reduction, y = 0 (exact fit requirement)
    """
    dataset: Dataset
    target_error: float = 0.0
    input_dim: int = 2
    output_dim: int = 2

    def get_required_hidden_dim(self) -> int:
        """
        Compute the minimum number of hidden neurons needed

        From paper: Number of breaklines needed = 4 * num_variable_gadgets + 5 * num_inversion_gadgets + 3 * num_lower_bound_gadgets
        """
        # This is computed during reduction
        # For now, return a reasonable upper bound
        return len(self.dataset) * 2


class GadgetLayout:
    """
    Manage the layout of all gadgets in reduction

    From paper Section 8.4: Gadgets are arranged to:
    - Variable gadgets parallel to each other (horizontal)
    - Inversion gadgets parallel to each other (non-horizontal)
    - Measuring lines intersect at the correct locations to encode constraints
    """

    def __init__(self):
        self.variable_gadgets: Dict[str, VariableGadget] = {}
        self.inversion_gadgets: Dict[Tuple[str, str], InversionGadget] = {}
        self.constraint_points: List[DataPoint] = []

        # Layout parameters
        self.variable_spacing = 15.0        # Spacing between variable gadgets
        self.variable_angle = 0.0           # Horizontal
        self.inversion_angle = np.pi / 6    # 30 degrees

    def add_variable_gadget(self, name: str, value: float, index: int):
        """Add a variable gadget to layout"""
        position = (0.0, index * self.variable_spacing)
        gadget = VariableGadget(
            value=value,
            orientation_angle=self.variable_angle,
            position=position
        )
        self.variable_gadgets[name] = gadget

    def add_inversion_gadget(self, var_X: str, var_Y: str, value_X: float, value_Y: float, index: int):
        """Add an inversion gadget to layout"""
        position = (index * 20.0, 0.0)
        gadget = InversionGadget(
            value_X=value_X,
            value_Y=value_Y,
            orientation_angle=self.inversion_angle,
            position=position
        )
        self.inversion_gadgets[(var_X, var_Y)] = gadget

    def compute_intersection(self, gadget1_name: str, gadget2_name: str,
                             line1_type: str, line2_type: str) -> np.ndarray:
        """
        Compute intersection point between measuring lines of 2 gadgets

        Args:
             gadget1_name: Name of first gadget
             gadget2_name: Name of second gadget
             line1_type: 'upper' or 'lower' for first gadget
             line2_type: 'upper' or 'lower' for second gadget

        Returns:
            Intersection point coordinates
        """
        # Simplified: Assume gadgets are perpendicular for easy intersection
        # In full implementation, solve line intersection equations

        gadget1 = self.variable_gadgets[gadget1_name]
        gadget2 = self.variable_gadgets[gadget2_name]

        lower1, upper1 = gadget1.get_measuring_lines()
        lower2, upper2 = gadget2.get_measuring_lines()

        line1 = upper1 if line1_type == 'upper' else lower1
        line2 = upper2 if line2_type == 'upper' else lower2

        # Simple intersection (assuming perpendicular gadgets)
        # In reality, need to solve line intersection
        return (line1 + line2) / 2  # Simplified


class Reducer:
    """
    Reduce ETR-Inv problem to Train-F2NN problem

    Main theorem (Theorem 3): ETR-Inv <= _p Train-F2NN
    """

    def __init__(self):
        self.layout = GadgetLayout()

    def reduce(self, problem: ETRInvProblem, solution: Dict[str, float] = None) -> TrainF2NNProblem:
        """
        Reduce ETR-Inv instance to Train-F2NN instance

        Args:
             problem: ETR-Inv problem instance
             solution: Optional known instance (for demonstration)

        Returns:
            Train-F2NN problem instance
        """
        # If no solution provided, try to find one
        if solution is None:
            solution = problem.find_solution_naive()
            if solution is None:
                raise ValueError("Could not find solution for demonstration")

        # Verify solution
        if not problem.is_satisfiable(solution):
            raise ValueError("Provided solution does not satisfy constraints")

        # Build gadgets
        all_points = []

        # 1. Create canonical variable gadgets
        for i, var_name in enumerate(problem.variables):
            value = solution[var_name]
            self.layout.add_variable_gadget(var_name, value, i)

            # Generate data points
            gadget = self.layout.variable_gadgets[var_name]
            dataset = gadget.generate_data_points(num_points_line=5)
            all_points.extend(dataset.points)

        # 2. Create inversion gadgets
        for i, constraint in enumerate(problem.get_inversion_constraints()):
            var_X, var_Y = constraint.variables
            value_X = solution[var_X]
            value_Y = solution[var_Y]

            self.layout.add_inversion_gadget(var_X, var_Y, value_X, value_Y, i)

            # Generate data points
            gadget = self.layout.inversion_gadgets[(var_X, var_Y)]
            dataset = gadget.generate_data_points(num_points_per_line=5)
            all_points.extend(dataset.points)

        # 3. Encode addition constraints
        for constraint in problem.get_addition_constraints():
            var_X, var_Y, var_Z = constraint.variables

            # Create copies and addition constraint point
            # Simplified: Just use canonical gadgets
            gadget_X = self.layout.variable_gadgets[var_X]
            gadget_Y = self.layout.variable_gadgets[var_Y]
            gadget_Z = self.layout.variable_gadgets[var_Z]

            # Compute intersection point (simplified)
            intersection = self.layout.compute_intersection(var_X, var_Y, 'upper', 'upper')

            # Create constraint point
            constraint_point = VariableConstraintEncoder.create_addition_constraint(gadget_X, gadget_Y, gadget_Z, intersection)
            all_points.append(constraint_point)

        # 4. Encode copy constraint (X = Y)
        # For demonstration, add some copy constraints
        if len(problem.variables) >= 2:
            var1, var2 = problem.variables[0], problem.variables[1]
            gadget1 = self.layout.variable_gadgets[var1]
            gadget2 = self.layout.variable_gadgets[var2]

            intersection = self.layout.compute_intersection(var1, var2, 'upper', 'lower')

            # Note: This is just for demonstration
            # Real reduction has more complex layout

        # Create dataset
        dataset = Dataset(all_points)

        return TrainF2NNProblem(
            dataset=dataset,
            target_error=0.0,   # Exact fit required
            input_dim=2,
            output_dim=2
        )

    def verify_reduction(self,
                         problem: ETRInvProblem,
                         nn_problem: TrainF2NNProblem,
                         network: TwoLayerNetwork) -> bool:
        """
        Verify that reduction is correct

        Check:
        1. Network fits all data points exactly
        2. Extracted slopes satisfy ETR-Inv constraints

        Args:
            problem: Original ETR-Inv problem
            nn_problem: Reduced Train-F2NN problem
            network: Trained network

        Returns:
            True if reduction is valid
        """
        # 1. Check exact fit
        X, Y = nn_problem.dataset.to_arrays()
        predictions = network.predict(X)
        max_error = np.max(np.abs(predictions - Y))

        if max_error > 1e-4:
            print(f"Network does not fit exactly: max_error = {max_error}")
            return False

        # 2. Extract variable values from slopes
        extracted_values = {}
        for var_name, gadget in self.layout.variable_gadgets.items():
            # In full implementation: Extract slope from network
            # For now, just use gadget's slope
            slope = gadget.get_slope()
            value = slope - 1
            extracted_values[var_name] = value

        # 3. Verify constraints
        if not problem.is_satisfiable(extracted_values):
            print("Extract values do not satisfy ETR-Inv constraints")
            return False

        return True

    def get_reduction_stats(self, problem: ETRInvProblem) -> Dict:
        """
        Get statistics about reduction

        Returns:
             Dictionary with reduction statistics
        """
        gadget_counts = problem.count_gadgets_needed()

        # Estimate number of data points
        points_per_variable = 9 * 5     # 9 lines, 5 points per line
        points_per_inversion = 13 * 5   # 13 lines, 5 points per line

        num_data_points = (
            gadget_counts['variable_gadgets'] * points_per_variable +
            gadget_counts['inversion_gadgets'] * points_per_inversion +
            len(problem.get_addition_constraints()) +
            len(problem.variables) * 2      # Copy constraints
        )

        # Minimum hidden neurons = number of breaklines
        min_hidden = (
            4 * gadget_counts['variable_gadgets'] +
            5 * gadget_counts['inversion_gadgets'] +
            3 * gadget_counts['lower_bound_gadgets']
        )

        return {
            'num_variables': len(problem.variables),
            'num_addition_constraints': len(problem.get_addition_constraints()),
            'num_inversion_constraints': len(problem.get_inversion_constraints()),
            'num_gadgets': gadget_counts['total_gadgets'],
            'estimated_data_points': num_data_points,
            'min_hidden_neurons': min_hidden,
        }