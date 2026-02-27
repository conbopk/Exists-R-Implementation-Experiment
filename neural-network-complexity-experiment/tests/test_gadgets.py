"""
Unit Tests for Gadgets

Test variable gadgets and inversion gadgets
"""
import pytest
import numpy as np

from src.gadgets.variable import VariableGadget, VariableConstraintEncoder
from src.gadgets.inversion import InversionGadget


class TestVariableGadget:
    """Test for VariableGadget"""

    def test_initialization(self):
        """Test gadget initialization"""
        gadget = VariableGadget(value=1.0)
        assert gadget.value == 1.0
        assert gadget.slope == 2.0  # s_X = X + 1

    def test_value_bounds(self):
        """Test value must be in [0.5, 2.0]"""
        # Valid values
        VariableGadget(value=0.5)
        VariableGadget(value=1.0)
        VariableGadget(value=2.0)

        # Invalid values
        with pytest.raises(ValueError):
            VariableGadget(value=0.3)

        with pytest.raises(ValueError):
            VariableGadget(value=2.5)

    def test_slope_calculation(self):
        """Test slope calculation s_X = X + 1"""
        test_values = [0.5, 1.0, 1.5, 2.0]
        for value in test_values:
            gadget = VariableGadget(value=value)
            assert gadget.get_slope() == value + 1.0

    def test_data_point_generation(self):
        """Test data point generation"""
        gadget = VariableGadget(value=1.0)
        dataset = gadget.generate_data_points(num_points_line=5)

        # Should have 9 lines x 5 points = 45 points
        assert len(dataset) == 9 * 5

        # Check all points have correct dimensions
        for point in dataset.points:
            assert point.x.shape == (2,)
            assert point.y.shape == (2,)

    def test_measuring_lines(self):
        """Test measuring line positions"""
        gadget = VariableGadget(value=1.0, orientation_angle=0.0)
        lower_line, upper_line = gadget.get_measuring_lines()

        # Lines should be perpendicular to orientation
        # and distance 1 from l4
        assert lower_line.shape == (2,)
        assert upper_line.shape == (2,)

    def test_expected_contributions(self):
        """Test expected contributions at measuring lines"""
        gadget = VariableGadget(value=1.0)
        slope = gadget.slope # 2.0

        # Lower measuring line: 3 - s_X
        assert gadget.expected_contribution('lower') == 3 - slope

        # Upper measuring line: 3 + s_X
        assert gadget.expected_contribution('upper') == 3 + slope

        # Center: 3
        assert gadget.expected_contribution('center') == 3

    def test_cross_section(self):
        """Test cross-section computation"""
        gadget = VariableGadget(value=1.0)

        # Test at data line positions
        positions = np.array(gadget.distances)
        values = gadget.compute_cross_section(positions)

        # Check values match expected labels
        for i, (pos, expected_label) in enumerate(zip(positions, gadget.labels)):
            assert np.isclose(values[i], expected_label, atol=0.1)


class TestInversionGadget:
    """Tests for InversionGadget"""

    def test_initialization(self):
        """Test gadget initialization"""
        X, Y = 1.5, 2.0/3.0
        gadget = InversionGadget(value_X=X, value_Y=Y)

        assert gadget.value_X == X
        assert gadget.value_Y == Y
        assert np.isclose(X * Y, 1.0)

    def test_inversion_constraint(self):
        """Test that X·Y = 1 is enforced"""
        # Valid pairs
        valid_pairs = [
            (1.0, 1.0),
            (1.5, 2.0/3.0),
            (2.0, 0.5),
            (0.5, 2.0),
        ]

        for X, Y in valid_pairs:
            gadget = InversionGadget(value_X=X, value_Y=Y)
            assert gadget.verify_inversion_constraint()

        # Invalid pair should raise error
        with pytest.raises(ValueError):
            InversionGadget(value_X=1.5, value_Y=0.5)

    def test_data_point_generation(self):
        """Test data point generation"""
        gadget = InversionGadget(value_X=1.0, value_Y=1.0)
        dataset = gadget.generate_data_points(num_points_per_line=5)

        # Should have 13 lines x 5 points = 65 points
        assert len(dataset) == 13 * 5

        # Check dimensions
        for point in dataset.points:
            assert point.x.shape == (2,)
            assert point.y.shape == (2,)    # 2 output dimensions

    def test_different_output_dimensions(self):
        """Test that two dimensions have different labels"""
        gadget = InversionGadget(value_X=1.5, value_Y=2.0/3.0)

        # Labels should be different for two dimensions
        assert gadget.labels_dim1 != gadget.labels_dim2

        # But should have same structure (shifted)
        assert len(gadget.labels_dim1) == len(gadget.labels_dim2)

    def test_cross_sections(self):
        """Test cross-section computation for both dimensions"""
        gadget = InversionGadget(value_X=1.5, value_Y=2.0/3.0)

        positions = np.array(gadget.distances)

        # Dimension 1
        values_dim1 = gadget.compute_cross_section(positions, dimension=0)
        for i, (pos, expected) in enumerate(zip(positions, gadget.labels_dim1)):
            if expected > 0:
                assert values_dim1[i] > 0

        # Dimension 2
        values_dim2 = gadget.compute_cross_section(positions, dimension=1)
        for i, (pos, expected) in enumerate(zip(positions, gadget.labels_dim2)):
            if expected > 0:
                assert values_dim2[i] > 0


class TestConstraintEncoders:
    """Test for constraint encoders"""

    def test_copy_constraint(self):
        """Test copy constraint X = Y"""
        gadget_A = VariableGadget(value=1.0)
        gadget_B = VariableGadget(value=1.0)

        intersection = np.array([0.0, 0.0])
        point = VariableConstraintEncoder.create_copy_constraint(gadget_A, gadget_B, intersection)

        # Label should be 6 (from paper)
        assert np.all(point.y == 6)
        assert point.label == "copy_constraint"

    def test_addition_constraint(self):
        """Test addition constraint X + Y = Z"""
        gadget_X = VariableGadget(value=0.5)
        gadget_Y = VariableGadget(value=1.0)
        gadget_Z = VariableGadget(value=1.5)

        intersection = np.array([0.0, 0.0])
        point = VariableConstraintEncoder.create_addition_constraint(gadget_X, gadget_Y, gadget_Z, intersection)

        # Label should be 10 (from paper)
        assert np.all(point.y == 10)
        assert point.label == "addition_constraint"


if __name__=="__main__":
    pytest.main([__file__, '-v'])

