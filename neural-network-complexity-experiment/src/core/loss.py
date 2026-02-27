"""
Loss Functions Module

Implement loss functions for training neural networks
"""
import numpy as np
from typing import Tuple, Optional


class LossFunction:
    """Base class for loss functions"""

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute loss value

        Args:
             predictions: Network predictions, shape (n_samples, output_dim)
             targets: Target values, shape (n_samples, output_dim)

        Returns:
            Loss value (scalar)
        """
        raise NotImplementedError

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. predictions

        Args:
            predictions: Network predictions, shape (n_samples, output_dim)
            targets: Target values, shape (n_samples, output_dim)

        Returns:
            Gradients, shape (n_samples, output_dim)
        """
        raise NotImplementedError


class MSELoss(LossFunction):
    """
    Mean Squared Error Loss

    L = (1/n) * sum_{i=1}^n ||y_i - f(x_i)||^2

    This is the main loss function used in the paper
    """

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute MSE loss

        Args:
            predictions: Network predictions, shape (n_samples, output_dim)
            targets: Target values, shape (n_samples, output_dim)

        Returns:
            MSE loss value
        """
        per_sample = np.sum((predictions - targets) ** 2, axis=1)
        return float(np.mean(per_sample))

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient: dL/dy = 2(y - t) / n

        Args:
             predictions: Network predictions, shape (n_samples, output_dim)
             targets: Target values, shape (n_samples, output_dim)

        Returns:
            Gradients w.r.t predictions
        """
        n_samples = predictions.shape[0]
        return 2 * (predictions - targets) / n_samples

    def compute_per_sample(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute loss per sample (useful for debugging)

        Args:
            predictions: Network predictions, shape (n_samples, output_dim)
            targets: Target values, shape (n_samples, output_dim)

        Returns:
            Loss per sample, shape (n_samples,)
        """
        return np.sum((predictions - targets) ** 2, axis=1)


class ExactFitLoss(MSELoss):
    """
    Exact Fit Loss - Check if the network fits exactly data points

    In paper, we need target error y = 0, which means fit exactly
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: Tolerance to be considered as "exact fit"
        """
        self.tolerance = tolerance

    def is_exact_fit(self, predictions: np.ndarray, targets: np.ndarray) -> bool:
        """
        Check if the network fits all data points exactly

        Args:
            predictions: Network predictions
            targets: Target values

        Returns:
            True if max error < tolerance
        """
        max_error = np.max(np.abs(predictions - targets))
        return max_error < self.tolerance

    def compute_max_error(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute maximum absolute error across all data points

        Args:
             predictions: Network predictions
             targets: Target values

        Returns:
            Maximum absolute error
        """
        return np.max(np.abs(predictions - targets))

    def compute_error_stats(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        """
        Compute comprehensive error statistics

        Args:
            predictions: Network predictions
            targets: Target values

        Returns:
            Dictionary with error statistics
        """
        errors = np.abs(predictions - targets)

        return {
            'mse': self.compute(predictions, targets),
            'mae': np.mean(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'is_exact_fit': self.is_exact_fit(predictions, targets),
        }


class DataPoint:
    """
    Represent a data point in the Train-F2NN problem

    In paper: data point is (x, y) with x ∈ R^2 và y ∈ R^2
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, label: Optional[str] = None):
        """
        Args:
            x: Input coordinates, shape (input_dim,)
            y: Target output, shape (output_dim,)
            label: Optional label for data point (useful for gadgets)
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.label = label

    def __repr__(self) -> str:
        label_str = f", label='{self.label}'" if self.label else ""
        return f"DataPoint(x={self.x}, y={self.y}{label_str})"


class Dataset:
    """
    Collection of data points for Train-F2NN problem
    """

    def __init__(self, points: list[DataPoint]):
        """
        Args:
            points: List of DataPoint objects
        """
        self.points = points
        self._validate()

    def _validate(self):
        """Validate that all points have consistent dimensions"""
        if not self.points:
            return

        input_dim = len(self.points[0].x)
        output_dim = len(self.points[0].y)

        for point in self.points:
            if len(point.x) != input_dim:
                raise ValueError(f"Inconsistent input dimensions: expected {input_dim}, got {len(point.x)}")
            if len(point.y) != output_dim:
                raise ValueError(f"Inconsistent output dimensions: expected {output_dim}, got {len(point.y)}")

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to numpy arrays

        Returns:
            X: Input array, shape (n_points, input_dim)
            Y: Output array, shape (n_points, output_dim)
        """
        X = np.array([point.x for point in self.points])
        Y = np.array([point.y for point in self.points])
        return X, Y

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> DataPoint:
        return self.points[idx]

    def __repr__(self) -> str:
        input_dim = len(self.points[0].x) if self.points else 0
        output_dim = len(self.points[0].y) if self.points else 0
        return f"Dataset(n_points={len(self)}, input_dim={input_dim}, output_dim={output_dim})"

    def add_point(self, point: DataPoint):
        """Add a new data point"""
        self.points.append(point)
        self._validate()

    def filter_by_label(self, label: str) -> "Dataset":
        """
        Filter data points by label

        Args:
             label: Label to filter by

        Returns:
            New Dataset with only matching points
        """
        filtered_points = [p for p in self.points if p.label == label]
        return Dataset(filtered_points)
