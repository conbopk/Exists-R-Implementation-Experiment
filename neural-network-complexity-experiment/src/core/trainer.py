"""
Training Module

Implement training algorithm for neural networks
"""
import numpy as np
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
import time
from tqdm import tqdm

from src.core.loss import MSELoss, ExactFitLoss, Dataset
from src.core.network import TwoLayerNetwork, NetworkWeights


@dataclass
class TrainingHistory:
    """Store training history"""
    losses: List[float]
    max_errors: List[float]
    iterations: List[int]
    times: List[float]

    def __init__(self):
        self.losses = []
        self.max_errors = []
        self.iterations = []
        self.times = []

    def add(self, iteration: int, loss: float, max_error: float, time: float):
        """Add a training step to history"""
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.max_errors.append(max_error)
        self.times.append(time)

    def get_best(self) -> Dict:
        """Get best result during training"""
        if not self.losses:
            return {}

        best_idx = np.argmin(self.losses)
        return {
            'iteration': self.iterations[best_idx],
            'loss': self.losses[best_idx],
            'max_error': self.max_errors[best_idx],
            'time': self.times[best_idx],
        }


class Trainer:
    """
    Train 2-layer neural networks

    Implement gradient descent with difference optimizers
    """

    def __init__(self,
                 network: TwoLayerNetwork,
                 learning_rate: float = 0.001,
                 optimizer: str = "adam",
                 tolerance: float = 1e-6,
                 max_iterations: int = 10000,
                 verbose: bool = True):
        """
        Args:
            network: Network to train
            learning_rate: Learning rate
            optimizer: Optimizer type ('sgd', 'adam')
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            verbose: Print progress
        """
        self.network = network
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.loss_fn = MSELoss()
        self.exact_fit_loss = ExactFitLoss(tolerance=tolerance)

        # Optimizer state
        self._init_optimizer_state()

    def _init_optimizer_state(self):
        """Initialize optimizer-specific state"""
        if self.optimizer == "adam":
            # Adam optimizer state
            self.m = {
                'W1': np.zeros_like(self.network.weights.W1),
                'b1': np.zeros_like(self.network.weights.b1),
                'W2': np.zeros_like(self.network.weights.W2),
                'b2': np.zeros_like(self.network.weights.b2),
            }
            self.v = {
                'W1': np.zeros_like(self.network.weights.W1),
                'b1': np.zeros_like(self.network.weights.b1),
                'W2': np.zeros_like(self.network.weights.W2),
                'b2': np.zeros_like(self.network.weights.b2),
            }
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0

    def train(self, dataset: Dataset, callback: Optional[Callable] = None) -> TrainingHistory:
        """
        Train network on dataset

        Args:
            dataset: Training dataset
            callback: Optional callback function called each iteration

        Returns:
            Training history
        """
        X, Y = dataset.to_arrays()
        history = TrainingHistory()
        start_time = time.time()

        iterator = tqdm(range(self.max_iterations)) if self.verbose else range(self.max_iterations)

        for iteration in iterator:
            # Forward pass
            predictions, _ = self.network.forward(X)

            # Compute loss
            loss = self.loss_fn.compute(predictions, Y)
            max_error = self.exact_fit_loss.compute_max_error(predictions, Y)

            # Log
            current_time = time.time() - start_time
            history.add(iteration, loss, max_error, current_time)

            if self.verbose and iteration % 100 == 0:
                iterator.set_description(f"Loss: {loss:.6f}, Max Error: {max_error:.6f}")

            # Check convergence
            if max_error < self.tolerance:
                if self.verbose:
                    print(f"\nConverged at iteration {iteration}")
                break

            # Compute gradients
            gradients = self.network.compute_gradients(X, Y)

            # Update weights
            self._update_weights(gradients)

            # Callback
            if callback is not None:
                callback(iteration, self.network, history)

        return history

    def _update_weights(self, gradients: NetworkWeights):
        """Update weights using selected optimizer"""
        if self.optimizer == "sgd":
            self._sgd_update(gradients)
        elif self.optimizer == "adam":
            self._adam_update(gradients)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def _sgd_update(self, gradients: NetworkWeights):
        """Vanilla SGD update"""
        self.network.weights.W1 -= self.learning_rate * gradients.W1
        self.network.weights.b1 -= self.learning_rate * gradients.b1
        self.network.weights.W2 -= self.learning_rate * gradients.W2
        self.network.weights.b2 -= self.learning_rate * gradients.b2

    def _adam_update(self, gradients: NetworkWeights):
        """Adam optimizer update"""
        self.t += 1

        # Update biased first moment estimate
        self.m['W1'] = self.beta1 * self.m['W1'] + (1 - self.beta1) * gradients.W1
        self.m['b1'] = self.beta1 * self.m['b1'] + (1 - self.beta1) * gradients.b1
        self.m['W2'] = self.beta1 * self.m['W2'] + (1 - self.beta1) * gradients.W2
        self.m['b2'] = self.beta1 * self.m['b2'] + (1 - self.beta1) * gradients.b2

        # Update biased second raw moment estimate
        self.v['W1'] = self.beta2 * self.v['W1'] + (1 - self.beta2) * (gradients.W1 ** 2)
        self.v['b1'] = self.beta2 * self.v['b1'] + (1 - self.beta2) * (gradients.b1 ** 2)
        self.v['W2'] = self.beta2 * self.v['W2'] + (1 - self.beta2) * (gradients.W2 ** 2)
        self.v['b2'] = self.beta2 * self.v['b2'] + (1 - self.beta2) * (gradients.b2 ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = {
            'W1': self.m['W1'] / (1 - self.beta1 ** self.t),
            'b1': self.m['b1'] / (1 - self.beta1 ** self.t),
            'W2': self.m['W2'] / (1 - self.beta1 ** self.t),
            'b2': self.m['b2'] / (1 - self.beta1 ** self.t)
        }

        # Compute bias-corrected second raw moment estimate
        v_hat = {
            'W1': self.v['W1'] / (1 - self.beta2 ** self.t),
            'b1': self.v['b1'] / (1 - self.beta2 ** self.t),
            'W2': self.v['W2'] / (1 - self.beta2 ** self.t),
            'b2': self.v['b2'] / (1 - self.beta2 ** self.t)
        }

        # Update parameters
        self.network.weights.W1 -= self.learning_rate * m_hat['W1'] / (np.sqrt(v_hat['W1']) + self.epsilon)
        self.network.weights.b1 -= self.learning_rate * m_hat['b1'] / (np.sqrt(v_hat['b1']) + self.epsilon)
        self.network.weights.W2 -= self.learning_rate * m_hat['W2'] / (np.sqrt(v_hat['W2']) + self.epsilon)
        self.network.weights.b2 -= self.learning_rate * m_hat['b2'] / (np.sqrt(v_hat['b2']) + self.epsilon)

    def evaluate(self, dataset: Dataset) -> Dict:
        """
        Evaluate network on dataset

        Args:
            dataset: Test dataset

        Returns:
            Dictionary with evaluation metrics
        """
        X, Y = dataset.to_arrays()
        predictions, _ = self.network.forward(X)

        return self.exact_fit_loss.compute_error_stats(predictions, Y)


class MultiStartTrainer:
    """
    Train with multiple random initializations

    Since Train-F2NN is non-convex, multiple starts can help find better solutions
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_starts: int = 10,
                 **trainer_kwargs):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            n_starts: Number of random starts
            **trainer_kwargs: Arguments passed to Trainer
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_starts = n_starts
        self.trainer_kwargs = trainer_kwargs

        self.best_network = None
        self.best_loss = float('inf')
        self.all_histories = []

    def train(self, dataset: Dataset) -> Dict:
        """
        Train with multiple random initializations

        Args:
            dataset: Training dataset

        Returns:
            Dictionary with best results and all histories
        """
        print(f"Training with {self.n_starts} random initializations...")

        for i in range(self.n_starts):
            print(f"\n--- Start {i+1}/{self.n_starts}")

            # Create new network with random initialization
            network = TwoLayerNetwork(self.input_dim, self.hidden_dim, self.output_dim)

            # Train
            trainer = Trainer(network, **self.trainer_kwargs)
            history = trainer.train(dataset)

            # Check if this is best so far
            final_loss = history.losses[-1]
            if final_loss < self.best_loss:
                self.best_loss = final_loss
                self.best_network = network
                print(f"New best loss: {final_loss:.6f}")

            self.all_histories.append(history)

        return {
            'best_network': self.best_network,
            'best_loss': self.best_loss,
            'all_histories': self.all_histories,
        }