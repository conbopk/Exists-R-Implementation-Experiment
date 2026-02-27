"""
Neural Network Architecture Module

Implement 2-layer fully connected neural network with ReLU activation
according to the paper "Training Fully Connected Neural Network is ∃R-Complete"
"""
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class NetworkWeights:
    """Container for weights and biases of network"""
    W1: np.ndarray  # Shape: (input_dim, hidden_dim)
    b1: np.ndarray  # Shape: (hidden_dim,)
    W2: np.ndarray  # Shape: (hidden_dim, output_dim)
    b2: np.ndarray  # Shape: (output_dim,)

    def to_vector(self) -> np.ndarray:
        """Flatten all weights into a single 1D vector."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])

    @classmethod
    def from_vector(cls, vector: np.ndarray, input_dim: int, hidden_dim: int, output_dim: int) -> "NetworkWeights":
        """Reconstruct weights from a 1D parameter vector."""
        idx = 0

        # W1
        W1_size = input_dim * hidden_dim
        W1 = vector[idx:idx + W1_size].reshape(input_dim, hidden_dim)
        idx += W1_size

        # b1
        b1 = vector[idx:idx + hidden_dim]
        idx += hidden_dim

        # W2
        W2_size = hidden_dim * output_dim
        W2 = vector[idx:idx + W2_size].reshape(hidden_dim, output_dim)
        idx += W2_size

        # b2
        b2 = vector[idx:idx + output_dim]

        return cls(W1=W1, b1=b1, W2=W2, b2=b2)

    def copy(self) -> "NetworkWeights":
        """Create a deep copy of the weights."""
        return NetworkWeights(
            W1=self.W1.copy(),
            b1=self.b1.copy(),
            W2=self.W2.copy(),
            b2=self.b2.copy()
        )


class TwoLayerNetwork:
    """
    2-layer Fully Connected Neural Network with ReLU activation.

    Architecture:
        Input (d_in) -> Hidden (d_h) -> Output (d_out)

    Forward pass:
        h = ReLU(W1^T * x + b1)
        y = W2^T * h + b2

    In the paper:
        - Input dimension = 2 (points in R^2)
        - Output dimension = 2 (to encode nonlinear constraints)
        - Hidden dimension = variable (number of neurons required)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
             input_dim: Number of input features (d_in)
             hidden_dim: Number of hidden neurons (d_h)
             output_dim: Number of output dimensions (d_out)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights randomly
        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> NetworkWeights:
        """Initialize weights using Xavier/He initialization."""
        W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        b1 = np.zeros(self.hidden_dim)
        W2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2.0 / self.hidden_dim)
        b2 = np.zeros(self.output_dim)

        return NetworkWeights(W1=W1, b1=b1, W2=W2, b2=b2)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform forward propagation.

        Args:
             X: Input data, shape (n_samples, input_dim)

        Returns:
            output: Network output, shape (n_samples, output_dim)
            hidden: Hidden layer activations (sau ReLU), shape (n_samples, hidden_dim)
        """
        # Hidden layer
        z1 = X @ self.weights.W1 + self.weights.b1  #(n, hidden_dim)
        hidden = np.maximum(0, z1)  # ReLU activation

        # Output layer
        output = hidden @ self.weights.W2 + self.weights.b2     # (n, output_dim)

        return output, hidden

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output cho input X

        Args:
             X: Input data, shape (n_samples, input_dim)

        Returns:
            Predictions, shape (n_samples, output_dim)
        """
        output, _ = self.forward(X)
        return output

    def compute_gradients(self, X: np.ndarray, Y: np.ndarray) -> NetworkWeights:
        """
        Compute gradients of MSE loss w.r.t. weights.

        Args:
            X: Input data, shape (n_samples, input_dim)
            Y: Target data, shape (n_samples, output_dim)

        Returns:
            Gradients w.r.t. each weight matrix
        """
        n_samples = X.shape[0]

        # Forward pass
        output, hidden = self.forward(X=X)

        # Compute loss gradient
        d_output = 2 * (output - Y) / n_samples # (n, output_dim)

        # Backprop through output layer
        d_W2 = hidden.T @ d_output          # (hidden_dim, output_dim)
        d_b2 = np.sum(d_output, axis=0)     # (output_dim,)

        # Backprop through ReLU
        d_hidden = d_output @ self.weights.W2.T     # (n, hidden_dim)
        z1 = X @ self.weights.W1 + self.weights.b1
        d_hidden[z1 <= 0] = 0       # ReLU gradient

        # Backprop through hidden layer
        d_W1 = X.T @ d_hidden   # (input_dim, hidden_dim)
        d_b1 = np.sum(d_hidden, axis=0)     # (hidden_dim,)

        return NetworkWeights(W1=d_W1, b1=d_b1, W2=d_W2, b2=d_b2)

    def get_breaklines(self) -> List[Tuple[np.ndarray, float]]:
        """
        Extract breaklines from the network.

        Breakline is hyperplane in input space where a ReLU neuron transfer from inactive -> active
        Each hidden neuron defines a breakline: w^T * x + b = 0

        Returns:
            List of (w, b) tuples, each tuple define a breakline
        """
        breaklines = []
        for i in range(self.hidden_dim):
            w = self.weights.W1[:, i]   # Weight vector cho neuron i
            b = self.weights.b1[i]      # Bias cho neuron i
            breaklines.append((w, b))

        return breaklines

    def set_weights(self, weights: NetworkWeights):
        """Set network weights."""
        self.weights = weights.copy()

    def get_weights(self) -> NetworkWeights:
        """Return a copy of network weights."""
        return self.weights.copy()

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return self.input_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim * self.output_dim + self.output_dim

    def __repr__(self) -> str:
        return (f"TwoLayerNetwork(input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
                f"params={self.count_parameters()})")


class PiecewiseLinearFunction:
    """
    Piecewise Linear Function generated by 2-layer ReLU network

    Implication in paper: Network output is piecewise linear function,
    with breaklines defined by ReLU neurons activate/deactivate.
    """

    def __init__(self, network: TwoLayerNetwork):
        """
        Args:
            network: Trained 2-layer network
        """
        self.network = network
        self.breaklines = network.get_breaklines()

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate function at given points.

        Args:
            points: Shape (n_points, input_dim)

        Returns:
            Function values, shape (n_points, output_dim)
        """
        return self.network.predict(points)

    def get_region(self, point: np.ndarray) -> List[bool]:
        """
        Determine activation pattern (region) for a given point.

        Args:
            point: Single point, shape (input_dim,)

        Returns:
            List of booleans indicating which neurons are active
        """
        z1 = point @ self.network.weights.W1 + self.network.weights.b1
        return [z > 0 for z in z1]

    def get_gradient(self, point: np.ndarray, output_dim: int = 0) -> np.ndarray:
        """
        Compute gradient of a specific output dimension at a given point.

        Args:
            point: Single point, shape (input_dim,)
            output_dim: Which output dimension to compute gradient for

        Returns:
            Gradient vector, shape (input_dim,)
        """
        # Xác định active neurons
        z1 = point @ self.network.weights.W1 + self.network.weights.b1
        active = z1 > 0

        # Gradient = sum of active neuron contributions
        gradient = np.zeros(self.network.input_dim)
        for i, is_active in enumerate(active):
            if is_active:
                w1_i = self.network.weights.W1[:, i]
                w2_i = self.network.weights.W2[i, output_dim]
                gradient += w1_i * w2_i

        return gradient