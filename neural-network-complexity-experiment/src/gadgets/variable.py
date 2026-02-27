"""
Variable Gadget Module

Implement Variable Gadget from paper to encode real variable X ∈ [1/2, 2]
"""
import numpy as np
from typing import Tuple

from src.core.loss import Dataset, DataPoint


class VariableGadget:
    """
    Variable Gadget encodes a real variable X ∈ [1/2, 2]

    From paper (Section 8.3.1):
    - Gadget consists of 9 parallel data lines l1, ..., l9
    - Slope s_X ∈ [3/2, 3] encodes value X = s_X - 1 ∈ [1/2, 2]
    - There are 4 breaklines: b1, b2, b3, b4 parallel to data lines
    - Function must fit exactly all data points

    Structure:
    - l1, l2, l3: Label 0 (flat region)
    - l4: Label 3 (central line)
    - l5: Label 6 (peak)
    - l6, l7, l8, l9: Labels decrease back to 0

    Cross-section through gadget forms piecewise linear function with slope s_X
    """

    def __init__(self,
                 value: float,
                 orientation_angle: float = 0.0,
                 position: Tuple[float, float] = (0.0, 0.0)):
        """
        Args:
            value: Variable value X ∈ [1/2, 2]
            orientation_angle: Gadget angle in R^2 (radians)
            position: Central location of gadget
        """
        if not (0.5 <= value <= 2.0):
            raise ValueError(f"Variable value must be in [0.5, 2.0], got {value}")

        self.value = value
        self.slope = value + 1.0 # s_X = X + 1
        self.orientation_angle = orientation_angle
        self.position = np.array(position)

        # Data line labels from paper
        self.labels = [0, 0, 0, 3, 6, 6, 6, 6, 0]

        # Relative distance from l1 (from paper Figure 8)
        self.distances = [0, 1, 2, 4, 7, 9, 10, 11, 13]

        # Measuring line distance
        self.measuring_line_distance = 1.0

    def get_slope(self) -> float:
        """Get the slope s_X of the variable gadget"""
        return self.slope

    def get_value(self) -> float:
        """Get the X of the variable"""
        return self.value

    def generate_data_points(self,
                             num_points_line: int = 5,
                             line_length: float = 10.0) -> Dataset:
        """
        Generate data points for variable gadget

        Args:
             num_points_line: Number of points per data line
             line_length: Length of each data line

        Returns:
            Dataset contains all data points
        """
        points = []

        # Direction vector of data lines
        direction = np.array([np.cos(self.orientation_angle), np.sin(self.orientation_angle)])

        # Perpendicular direction (for spacing between lines)
        perp_direction = np.array([-np.sin(self.orientation_angle), np.cos(self.orientation_angle)])

        # Generate points for each data line
        for i, (distance, label) in enumerate(zip(self.distances, self.labels)):
            # Position of line i
            line_center = self.position + perp_direction * distance

            # Generate points along the line
            for j in range(num_points_line):
                t = (j - num_points_line // 2) * (line_length / num_points_line)
                point_pos = line_center + direction * t

                # Target output (same in both dimensions for variable gadget)
                target = np.array([label, label], dtype=float)

                points.append(DataPoint(
                    x=point_pos,
                    y=target,
                    label=f"var_line_{i}"
                ))

        return Dataset(points)

    def get_measuring_lines(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get positions of upper and lower measuring lines

        Measuring lines are 1 unit away from l4

        Returns:
            (lower_line_center, upper_line_center)
        """
        perp_direction = np.array([-np.sin(self.orientation_angle), np.cos(self.orientation_angle)])

        # l4 is at distance 4 from l1
        l4_position = self.position + perp_direction * 4

        # Lower measuring line (towards l3)
        lower_line = l4_position - perp_direction * self.measuring_line_distance

        # Upper measuring line (towards l5)
        upper_line = l4_position + perp_direction * self.measuring_line_distance

        return lower_line, upper_line

    def expected_contribution(self, point_type: str) -> float:
        """
        Expected contribution of gadgets at measuring lines

        Args:
            point_type: 'lower' or 'upper' measuring line

        Returns:
            Expected function value
        """
        if point_type == 'lower':
            # Lower measuring line: f(p) = 3 - s_X
            return 3 - self.slope
        elif point_type == 'upper':
            # Upper measuring line: f(p) = 3 + s_X
            return 3 + self.slope
        elif point_type == 'center':
            # l4: f(p) = 3
            return 3
        else:
            raise ValueError(f"Unknown point type: {point_type}")

    def compute_cross_section(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute ideal cross-section of variable gadget

        Args:
             positions: Positions perpendicular to data lines

        Returns:
            Expected function values at these positions
        """
        values = np.zeros_like(positions, dtype=float)

        # Breaklines at distances: 1, 4, 7, 10
        b1, b2, b3, b4 = 1, 4, 7, 10

        for i, pos in enumerate(positions):
            if pos <= b1:
                values[i] = 0
            elif pos <= b2:
                # Sloped region 1: from 0 to 3 with slope s_X
                values[i] = (pos - b1) * self.slope
            elif pos <= b3:
                # Flat region at 6
                values[i] = 6
            elif pos <= b4:
                # Sloped region 2: from 6 to 3 with slope -s_X
                values[i] = 6 - (pos - b3) * self.slope
            else:
                values[i] = 0

        return values

    def __repr__(self) -> str:
        return (f"VariableGadget(value={self.value:.3f}, slope={self.slope:.3f}, "
                f"angle={np.degrees(self.orientation_angle):.1f}⁰)")

class VariableConstraintEncoder:
    """
    Encoder for linear constraints between variables

    From paper: Can encode X = Y (copy) and X + Y = Z (addition)
    """

    @staticmethod
    def create_copy_constraint(gadget_A: VariableGadget,
                               gadget_B: VariableGadget,
                               intersection_point: np.ndarray) -> DataPoint:
        """
        Create data point to enforce X = Y

        Args:
            gadget_A: Variable gadget for X
            gadget_B: Variable gadget for Y
            intersection_point: Point lies on upper measuring line of A and lower measuring line of B

        Returns:
            Datapoint with label 6 (from paper: 4|A| + 2|B| = 4*1 + 2*1 = 6)
        """
        # Expected contribution
        # A contributes 3 + s_A (upper measuring line)
        # B contributes 3 - s_B (lower measuring line)
        # Total = 6 + s_A - s_B
        # Label = 6 enforces s_A = s_B, hence A = B

        label_value = 6
        target = np.array([label_value, label_value])

        return DataPoint(
            x=intersection_point,
            y=target,
            label="copy_constraint"
        )

    @staticmethod
    def create_addition_constraint(gadget_X: VariableGadget,
                                   gadget_Y: VariableGadget,
                                   gadget_Z: VariableGadget,
                                   intersection_point: np.ndarray) -> DataPoint:
        """
        Create data point to enforce X + Y = Z

        Args:
             gadget_X, gadget_Y: Upper measuring lines
             gadget_Z: Lower measuring line
             intersection_point: Point lies at intersection of 3 measuring lines

        Returns:
            DataPoint with label 10 (from paper: 4*2 + 2*1 = 10)
        """
        # Expected contribution:
        # X contributes 3 + s_X (upper)
        # Y contributes 3 + s_Y (upper)
        # Z contributes 3 - s_Z (lower)
        # Total = 9 + s_X + s_Y - s_Z
        # Label = 10 enforces s_X + s_Y = s_Z + 1
        # Since X = s_X - 1, Y = s_Y - 1, Z = s_Z - 1:
        # X + Y = (s_X - 1) + (s_Y - 1) = s_X + s_Y - 2
        # Z = s_Z - 1
        # Constraint becomes: X + Y = Z

        label_value = 10
        target = np.array([label_value, label_value])

        return DataPoint(
            x=intersection_point,
            y=target,
            label="addition_constraint"
        )