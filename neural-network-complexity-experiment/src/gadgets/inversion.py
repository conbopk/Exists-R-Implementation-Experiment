"""
Inversion Gadget Module

Implement Inversion Gadget from paper to encode constraint nonlinear X.Y = 1
"""
import numpy as np
from typing import Tuple

from src.core.loss import Dataset, DataPoint
from src.gadgets.variable import VariableGadget


class InversionGadget:
    """
    Inversion Gadget encode constraint nonlinear X.Y = 1

    From paper (section 8.3.4):
    - Gadget consists of 13 parallel data lines l1, ..., l13
    - Uses 2 output dimensions to encode 2 variables simultaneously
    - Has 5 breaklines to enforce nonlinear relationship
    - Labels differ in 2 dimensions

    Key insight:
    - Dimension 1 encodes X with slope s_X
    - Dimension 2 encodes Y with slope s_Y
    - Constraint from geometry: s_X · s_Y = constraint
    - From there: X · Y = (s_X - 1)(s_Y - 1) = 1

    Structure (from paper):
    Line   | Distance to ℓ1 | Label dim 1 | Label dim 2
    -------|-----------------|-------------|-------------
    ℓ1     | 0               | 0           | 0
    ℓ2     | 1               | 0           | 0
    ℓ3     | 2               | 0           | 0
    ℓ4     | 4               | 3           | 0
    ℓ5     | 7               | 6           | 3
    ℓ6     | 9               | 6           | 6
    ℓ7     | 10              | 6           | 6
    ℓ8     | 11              | 6           | 6
    ℓ9     | 13              | 4           | 4
    ℓ10    | 15              | 2           | 2
    ℓ11    | 17              | 0           | 0
    ℓ12    | 18              | 0           | 0
    ℓ13    | 19              | 0           | 0
    """

    def __init__(self,
                 value_X: float,
                 value_Y: float,
                 orientation_angle: float = 0.0,
                 position: Tuple[float, float] = (0.0, 0.0)):
        """
        Args:
            value_X: value of variable X ∈ [1/2, 2]
            value_Y: value of variable Y ∈ [1/2, 2]
            orientation_angle: gadget angle in R^2 (radians)
            position: central position of gadget
        """
        if not (0.5 <= value_X <= 2.0):
            raise ValueError(f"X must be in [0.5, 2.0], got {value_X}")
        if not (0.5 <= value_Y <= 2.0):
            raise ValueError(f"Y must be in [0.5, 2.0], got {value_Y}")

        # Check inversion constraint
        self.value_X = value_X
        self.value_Y = value_Y
        self.slope_X = value_X + 1.0    # s_X = X + 1
        self.slope_Y = value_Y + 1.0    # s_Y = Y + 1
        self.orientation_angle = orientation_angle
        self.position = np.array(position)

        # Data line structure from paper
        self.distances = [0, 1, 2, 4, 7, 9, 10, 11, 13, 15, 17, 18, 19]

        # Labels for dimension 1 (encoding X)
        self.labels_dim1 = [0, 0, 0, 3, 6, 6, 6, 6, 4, 2, 0, 0, 0]

        # Labels for dimension 2 (encoding Y)
        self.labels_dim2 = [0, 0, 0, 0, 3, 6, 6, 6, 4, 2, 0, 0, 0]

    def get_slopes(self) -> Tuple[float, float]:
        """Get slopes (s_X, s_Y) of inversion gadget"""
        return self.slope_X, self.slope_Y

    def get_values(self) -> Tuple[float, float]:
        """Get values (X, Y) of two variable"""
        return self.value_X, self.value_Y

    def generate_data_points(self,
                             num_points_per_line: int = 5,
                             line_length: float = 10.0) -> Dataset:
        """
        Generate data points for inversion gadget

        Args:
             num_points_per_line: Number of points per data line
             line_length: Length of each data line

        Returns:
            Dataset constrains all data points
        """
        points = []

        # Direction vector of data lines
        direction = np.array([np.cos(self.orientation_angle), np.sin(self.orientation_angle)])

        # Perpendicular
        perp_direction = np.array([-np.sin(self.orientation_angle), np.cos(self.orientation_angle)])

        # Generate points for each data line
        for i, (distance, label1, label2) in enumerate(zip(self.distances, self.labels_dim1, self.labels_dim2)):
            # Position of line i
            line_center = self.position + perp_direction * distance

            # Generate points along the line
            for j in range(num_points_per_line):
                t = (j - num_points_per_line // 2) * (line_length / num_points_per_line)
                point_pos = line_center + direction * t

                # Target output (different for each dimension)
                target = np.array([label1, label2], dtype=float)

                points.append(DataPoint(
                    x=point_pos,
                    y=target,
                    label=f"inv_line_{i}"
                ))

        return Dataset(points)

    def get_dimension_measuring_lines(self, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get measuring lines for a specific dimension

        Args:
            dimension: 0 for X (dimension 1), 1 for Y (dimension 2)

        Returns:
            (lower_line_center, upper_line_center)
        """
        perp_direction = np.array([-np.sin(self.orientation_angle), np.cos(self.orientation_angle)])

        if dimension == 0:  # Dimension 1 (X)
            # l4 is the reference line for dimension 1
            l4_position = self.position + perp_direction * 4
            lower_line = l4_position - perp_direction * 1.0
            upper_line = l4_position + perp_direction * 1.0
        else:   # dimension 2 (Y)
            # l5 is the reference line for dimension 2
            l5_position = self.position + perp_direction * 7
            lower_line = l5_position - perp_direction * 1.0
            upper_line = l5_position + perp_direction * 1.0

        return lower_line, upper_line

    def compute_cross_section(self, positions: np.ndarray, dimension: int) -> np.ndarray:
        """
        Compute ideal cross-section for a dimension

        Args:
            positions: Positions perpendicular to data lines
            dimension: 0 for dimension 1, 1 for dimension 2

        Returns:
            Expected function values at these positions
        """
        values = np.zeros_like(positions)

        if dimension == 0: # Dimension 1
            # Breaklines for dimension 1
            b1, b2, b3, b4, b5 = 2, 4, 9, 13, 17
            slope = self.slope_X

            for i, pos in enumerate(positions):
                if pos <= b1:
                    values[i] = 0
                elif pos <= b2:
                    values[i] = (pos - b1) * slope
                elif pos <= b3:
                    values[i] = 6
                elif pos <= b4:
                    values[i] = 6 - (pos - b3) * slope / 2
                elif pos <= b5:
                    values[i] = 2 - (pos - b4) * slope / 2
                else:
                    values[i] = 0

        else: # Dimension 2
            # Breaklines for dimension 2
            b1, b2, b3, b4, b5 = 4, 7, 11, 15, 17
            slope = self.slope_Y

            for i, pos in enumerate(positions):
                if pos <= b1:
                    values[i] = 0
                elif pos <= b2:
                    values[i] = (pos - b1) * slope
                elif pos <= b3:
                    values[i] = 6
                elif pos <= b4:
                    values[i] = 6 - (pos - b3) * slope / 2
                elif pos <= b5:
                    values[i] = 2 - (pos - b4) * slope / 2
                else:
                    values[i] = 0

        return values

    def verify_inversion_constraint(self) -> bool:
        """
        Verify that the inversion constraint X·Y = 1 is satisfied

        Returns:
            True if constraint satisfied (within tolerance)
        """
        product = self.value_X * self.value_Y
        return np.isclose(product, 1.0, rtol=1e-3)

    def __repr__(self) -> str:
        product = self.value_X * self.value_Y
        return (f"InversionGadget(X={self.value_X:.3f}, Y={self.value_Y:.3f}, "
                f"X*Y={product:.3f}, "
                f"angle={np.degrees(self.orientation_angle):.1f}⁰)")

class InversionConstraintEncoder:
    """
    Encoder for inversion constraint X•Y = 1
    """

    @staticmethod
    def create_copy_to_inversion(variable_gadget_X: 'VariableGadget',
                                 inversion_gadget: InversionGadget,
                                 intersection_point: np.ndarray,
                                 target_dimension: int) -> DataPoint:

        """
        Create data point to copy value from variables gadget intp inversion gadget

        Args:
             variable_gadget_X: Variable gadget for X
             inversion_gadget: Inversion gadget
             intersection_point: Intersection point
             target_dimension: 0 for dimension 1, 1 for dimension 2

        Returns:
            DataPoint with appropriate label
        """
        # Use weak data points concept from paper
        # Label only active in target dimension

        if target_dimension == 0:
            # Active in dimension 1, >= 0 in dimension 2
            label = np.array([6, 0])
        else:
            # >= 0 in dimension 1, active in dimension 2
            label = np.array([0, 6])

        return DataPoint(
            x=intersection_point,
            y=label,
            label=f"copy_to_inv_dim{target_dimension}"
        )

    @staticmethod
    def verify_slopes_satisfy_inversion(slope_X: float, slope_Y: float) -> bool:
        """
        Verify that slopes satisfy inversion constraint

        From paper: (s_X - 1)(s_Y - 1) = 1

        Args:
            slope_X: Slope s_X
            slope_Y: Slope s_Y

        Returns:
            True if constraint satisfy
        """
        X = slope_X - 1
        Y = slope_Y - 1
        product = X * Y
        return np.isclose(product, 1.0, rtol=1e-3)