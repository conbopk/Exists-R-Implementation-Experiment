"""
Visualization Module

Visualize gadgets, networks, and experimental results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from pathlib import Path

from src.core.loss import Dataset
from src.core.network import TwoLayerNetwork
from src.core.trainer import TrainingHistory
from src.gadgets.inversion import InversionGadget
from src.gadgets.variable import VariableGadget

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class GadgetVisualizer:
    """Visualize gadgets"""

    @staticmethod
    def plot_variable_gadget(gadget: VariableGadget, save_path: Optional[Path] = None):
        """
        Plot variable gadget with cross-section view

        Args:
             gadget: VariableGadget to visualize
             save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left: Top-down view of data lines
        ax1.set_title(f"Variable Gadget - Top View\nValue X = {gadget.value:.3f}, Slope = {gadget.slope:.3f}")
        ax1.set_xlabel("Direction along data lines")
        ax1.set_ylabel("Perpendicular distance")
        ax1.grid(True, alpha=0.3)

        # Draw data lines
        line_length = 10.0
        direction = np.array([np.cos(gadget.orientation_angle), np.sin(gadget.orientation_angle)])
        perp_direction = np.array([-np.sin(gadget.orientation_angle), np.cos(gadget.orientation_angle)])
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(gadget.distances)))

        for i, (distance, label) in enumerate(zip(gadget.distances, gadget.labels)):
            line_center = gadget.position + perp_direction * distance
            start = line_center - direction * line_length / 2
            end = line_center + direction * line_length / 2

            ax1.plot([start[0], end[0]], [start[1], end[1]], color=colors[i], linewidth=2, label=f"l{i+1} (label={label})")

        # Draw measuring lines
        lower_line, upper_line = gadget.get_measuring_lines()
        start_lower = lower_line - direction * line_length / 2
        end_lower = lower_line + direction * line_length / 2
        start_upper = upper_line - direction * line_length / 2
        end_upper = upper_line + direction * line_length / 2

        ax1.plot([start_lower[0], end_lower[0]], [start_lower[1], end_lower[1]], 'r--', linewidth=2, label='Lower measuring line')
        ax1.plot([start_upper[0], end_upper[0]], [start_upper[1], end_upper[1]], 'b--', linewidth=2, label='Upper measuring line')

        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.axis('equal')

        # Right: Cross-section view
        ax2.set_title('Cross-section (Perpendicular to data lines)')
        ax2.set_xlabel("Distance from l1")
        ax2.set_ylabel("Function value")
        ax2.grid(True, alpha=0.3)

        # Plot ideal cross-section
        positions = np.linspace(0, 14, 1000)
        values = gadget.compute_cross_section(positions)
        ax2.plot(positions, values, 'b-', linewidth=2, label='Ideal function')

        # Mark data points
        for i, (distance, label) in enumerate(zip(gadget.distances, gadget.labels)):
            ax2.plot(distance, label, 'ro', markersize=8)
            ax2.text(distance, label + 0.3, f"l{i+1}", ha='center', fontsize=8)

        # Mark measuring lines
        lower_contrib = gadget.expected_contribution('lower')
        upper_contrib = gadget.expected_contribution('upper')

        ax2.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='Lower measuring')
        ax2.axvline(x=5, color='b', linestyle='--', alpha=0.5, label='Upper measuring')
        ax2.axhline(y=lower_contrib, color='r', linestyle=':', alpha=0.3)
        ax2.axhline(y=upper_contrib, color='b', linestyle=':', alpha=0.3)

        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_inversion_gadget(gadget: InversionGadget,
                              save_path: Optional[Path] = None):
        """
        Plot inversion gadget with both dimensions

        Args:
             gadget: InversionGadget to visualize
             save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Title
        product = gadget.value_X * gadget.value_Y
        fig.suptitle(f"Inversion Gadget: X={gadget.value_X:.3f}, Y={gadget.value_Y:.3f}, X·Y={product:.3f}", fontsize=14, fontweight='bold')

        # Top-down view (shared)
        ax_top = axes[0, 0]
        ax_top.set_title("Top-Down View")
        ax_top.set_xlabel("Direction along data lines")
        ax_top.set_ylabel("Perpendicular distance")
        ax_top.grid(True, alpha=0.3)

        line_length = 10.0
        direction = np.array([np.cos(gadget.orientation_angle), np.sin(gadget.orientation_angle)])
        perp_direction = np.array([-np.sin(gadget.orientation_angle), np.cos(gadget.orientation_angle)])

        # Draw data lines with different colors for different label combinations
        for i, distance in enumerate(gadget.distances):
            label1 = gadget.labels_dim1[i]
            label2 = gadget.labels_dim2[i]

            line_center = gadget.position + perp_direction * distance
            start = line_center - direction * line_length / 2
            end = line_center + direction * line_length / 2

            # Color based on labels
            if label1 == 0 and label2 == 0:
                color = 'gray'
            elif label1 > 0 and label2 == 0:
                color = 'red'
            elif label1 == 0 and label2 > 0:
                color = 'blue'
            else:
                color = 'purple'

            ax_top.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2, alpha=0.7, label=f"l{i+1} ({label1},{label2})")

        ax_top.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
        ax_top.axis('equal')

        # Cross-section dimension 1
        ax_dim1 = axes[0, 1]
        ax_dim1.set_title(f"Dimension 1 Cross-section (X = {gadget.value_X:.3f})")
        ax_dim1.set_xlabel(f"Distance from l1")
        ax_dim1.set_ylabel(f"Function value")
        ax_dim1.grid(True, alpha=0.3)

        positions = np.linspace(0, 20, 1000)
        values_dim1 = gadget.compute_cross_section(positions, dimension=0)
        ax_dim1.plot(positions, values_dim1, 'r-', linewidth=2, label='Dimension 1')

        # Mark data points
        for i, (distance, label) in enumerate(zip(gadget.distances, gadget.labels_dim1)):
            if label > 0:
                ax_dim1.plot(distance, label, 'ro', markersize=8)
                ax_dim1.text(distance, label + 0.3, f"l{i+1}", ha='center', fontsize=7)

        ax_dim1.legend()

        # Cross-section dimension 2
        ax_dim2 = axes[1, 0]
        ax_dim2.set_title(f"Dimension 2 Cross-section (Y = {gadget.value_Y:.3f})")
        ax_dim2.set_xlabel("Distance from l1")
        ax_dim2.set_ylabel("Function value")
        ax_dim2.grid(True, alpha=0.3)

        values_dim2 = gadget.compute_cross_section(positions, dimension=1)
        ax_dim2.plot(positions, values_dim2, 'b-', linewidth=2, label='Dimension 2')

        # Mark data points
        for i, (distance, label) in enumerate(zip(gadget.distances, gadget.labels_dim2)):
            if label > 0:
                ax_dim2.plot(distance, label, 'bo', markersize=8)
                ax_dim2.text(distance, label + 0.3, f"l{i+1}", ha='center', fontsize=7)

        ax_dim2.legend()

        # Both dimensions overlay
        ax_both = axes[1, 1]
        ax_both.set_title("Both dimensions Overlay")
        ax_both.set_xlabel("Distance from l1")
        ax_both.set_ylabel("Function value")
        ax_both.grid(True, alpha=0.3)

        ax_both.plot(positions, values_dim1, 'r-', linewidth=2, label=f'Dim 1 (X={gadget.value_X:.3f})', alpha=0.7)
        ax_both.plot(positions, values_dim2, 'b-', linewidth=2, label=f'Dim 2 (Y={gadget.value_Y:.3f})', alpha=0.7)
        ax_both.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class NetworkVisualizer:
    """Visualize neural networks"""

    @staticmethod
    def plot_training_history(history: TrainingHistory, save_path: Optional[Path] = None):
        """
        Plot training history

        Args:
             history: Training history object
             save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curve
        ax_loss = axes[0, 0]
        ax_loss.plot(history.iterations, history.losses, 'b-', linewidth=2)
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("MSE Loss")
        ax_loss.set_title("Training Loss")
        ax_loss.set_yscale("log")
        ax_loss.grid(True, alpha=0.3)

        # Max error curve
        ax_error = axes[0, 1]
        ax_error.plot(history.iterations, history.max_errors, 'r-', linewidth=2)
        ax_error.set_xlabel("Iteration")
        ax_error.set_ylabel("Max Absolute Error")
        ax_error.set_title("Maximum Error")
        ax_error.set_yscale("log")
        ax_error.grid(True, alpha=0.3)
        ax_error.axhline(y=1e-6, color='g', linestyle='--', label='Tolerance (1e-6)')
        ax_error.legend()

        # Time vs Loss
        ax_time = axes[1, 0]
        ax_time.plot(history.times, history.losses, 'g-', linewidth=2)
        ax_time.set_xlabel("Time (seconds)")
        ax_time.set_ylabel("MSE Loss")
        ax_time.set_title("Loss vs Time")
        ax_time.set_yscale("log")
        ax_time.grid(True, alpha=0.3)

        # Convergence summary
        ax_summary = axes[1, 1]
        ax_summary.axis('off')

        best = history.get_best()
        if best:
            summary_text = f"""
Training Summary:
-----------------
Best Iteration: {best['iteration']}
Final Loss: {history.losses[-1]:.6e}
Best Loss: {best['loss']:.6e}
Final Max Error: {history.max_errors[-1]:.6e}
Best Max Error: {best['max_error']:.6e}
Total Time: {history.times[-1]:.2f}s
Total Iterations: {len(history.iterations)}
"""
            ax_summary.text(0.1, 0.5, summary_text, fontsize=11,
                            verticalalignment='center', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_network_predictions(network: TwoLayerNetwork,
                                 dataset: Dataset,
                                 save_path: Optional[Path] = None):
        """
        Plot network predictions vs targets

        Args:
            network: Trained network
            dataset: Test dataset
            save_path: Optional path to save figure
        """
        X, Y = dataset.to_arrays()
        predictions = network.predict(X)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Dimension 1
        ax1 = axes[0]
        ax1.scatter(Y[:, 0], predictions[:, 0], alpha=0.6, s=50)
        ax1.plot([Y[:, 0].min(), Y[:, 0].max()], [Y[:, 0].min(), Y[:, 0].max()],
                    'r--', linewidth=2, label='Perfect fit')
        ax1.set_xlabel("Target (Dimension 1)")
        ax1.set_ylabel("Prediction (Dimension 1)")
        ax1.set_title("Dimension 1: Predictions vs Targets")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Dimension 2
        ax2 = axes[1]
        ax2.scatter(Y[:, 1], predictions[:, 1], alpha=0.6, s=50, color='orange')
        ax2.plot([Y[:, 1].min(), Y[:, 1].max()], [Y[:, 1].min(), Y[:, 1].max()],
                 'r--', linewidth=2, label='Perfect fit')
        ax2.set_xlabel("Target (Dimension 2)")
        ax2.set_ylabel("Prediction (Dimension 2)")
        ax2.set_title("Dimension 2: Predictions vs Targets")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_breaklines(network: TwoLayerNetwork,
                        xlim: Tuple[float, float] = (-10, 10),
                        ylim: Tuple[float, float] = (-10, 10),
                        save_path: Optional[Path] = None):
        """
        Plot breaklines of network in input space

        Args:
             network: Trained network
             xlim: X-axis limits
             ylim: Y-axis limits
             save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        breaklines = network.get_breaklines()

        for i, (w, b) in enumerate(breaklines):
            # Breakline: w^T * x + b = 0
            # For 2D: w1*x1 + w2*x2 + b = 0
            # => x2 = -(w1*x1 + b) / w2

            if abs(w[1]) > 1e-6:    # Non-vertical line
                x1 = np.array([xlim[0], xlim[1]])
                x2 = -(w[0] * x1 + b) / w[1]
                ax.plot(x1, x2, linewidth=2, label=f"Breakline {i+1}", alpha=0.7)
            else:   # Vertical line
                x1 = -b / w[0] if abs(w[0]) > 1e-6 else 0
                ax.axvline(x=x1, linewidth=2, label=f'Breakline {i+1}', alpha=0.7)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Network Breaklines (Total: {len(breaklines)})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()