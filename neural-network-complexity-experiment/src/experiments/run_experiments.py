"""
Main Experiment Script

Run experiments to demonstrate concepts from paper
"""
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.loss import Dataset
from src.core.network import TwoLayerNetwork
from src.core.trainer import Trainer
from src.gadgets.inversion import InversionGadget
from src.gadgets.variable import VariableGadget, VariableConstraintEncoder
from src.reduction.etr_inv import create_simple_problem
from src.reduction.reducer import Reducer
from src.utils.config import get_config, ProjectConfig
from src.utils.visualization import GadgetVisualizer, NetworkVisualizer


def experiment_1_variable_gadget():
    """
    Experiment 1: Test Variable Gadget

    Purpose:
    - Verify that the variable gadget works correctly
    - Visualize the structure of the gadget
    - Test that the network can fit the variable gadget
    """
    print("=" * 60)
    print("EXPERIMENT 1: Variable Gadget Test")
    print("=" * 60)

    config = get_config()

    # Create variable gadget with value X = 1.0
    print("\n1. Creating Variable Gadget with X = 1.0...")
    gadget = VariableGadget(value=1.0, orientation_angle=0.0)
    print(f"    {gadget}")
    print(f"    Expected contributions:")
    print(f"    - Lower measuring line: {gadget.expected_contribution('lower'):.3f}")
    print(f"    - Center (l4): {gadget.expected_contribution('center'):.3f}")
    print(f"    - Upper measuring line: {gadget.expected_contribution('upper'):.3f}")

    # Generate data points
    print("\n2. Generating data points...")
    dataset = gadget.generate_data_points(num_points_line=5)
    print(f"    Generated {len(dataset)} data points")

    # Visualize gadget
    print("\n3. Visualizing gadget...")
    GadgetVisualizer.plot_variable_gadget(
        gadget,
        save_path=config.plots_dir / "variable_gadget_X1.0.png"
    )

    # Train network to fit gadget
    print("\n4. Training network to fit gadget...")
    network = TwoLayerNetwork(
        input_dim=2,
        hidden_dim=4,   # Need 4 breaklines for variable gadget
        output_dim=2
    )

    trainer = Trainer(
        network=network,
        learning_rate=0.01,
        max_iterations=5000,
        tolerance=1e-4,
        verbose=True
    )

    history = trainer.train(dataset=dataset)

    # Visualize training
    print("\n5. Visualizing training results...")
    NetworkVisualizer.plot_training_history(
        history,
        save_path=config.plots_dir / "variable_gadget_training.png"
    )

    # Evaluate
    results = trainer.evaluate(dataset)
    print("\n6. Evaluation Results:")
    print(f"   MSE Loss: {results['mse']:.6e}")
    print(f"   Max Error: {results['max_error']:.6e}")
    print(f"   Is Exact Fit: {results['is_exact_fit']}")

    return gadget, network, history


def experiment_2_inversion_gadget():
    """
    Experiment 2: Test Inversion Gadget

    Purpose:
    - Verify nonlinear constraint X·Y = 1
    - Visualize inversion gadget with 2 dimensions
    - Test network training
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Inversion Gadget Test")
    print("=" * 60)

    config = get_config()

    # Create inversion gadget with X = 1.5, Y = 2/3
    print("\n1. Creating Inversion Gadget...")
    X, Y = 1.5, 2.0/3.0
    print(f"    X = {X:.4f}, Y = {Y:.4f}, X*Y = {X*Y:.4f}")

    gadget = InversionGadget(
        value_X=X,
        value_Y=Y,
        orientation_angle=np.pi / 6
    )
    print(f"    {gadget}")
    print(f"    Constraint satisfied: {gadget.verify_inversion_constraint()}")

    # Generate data points
    print("\n2. Generating data point...")
    dataset = gadget.generate_data_points(num_points_per_line=5)
    print(f'    Generated {len(dataset)} data points')

    # Visualize gadget
    print("\n3. Visualizing gadget...")
    GadgetVisualizer.plot_inversion_gadget(
        gadget,
        save_path=config.plots_dir / "inversion_gadget.png"
    )

    # Train network
    print("\n4. Training network...")
    network = TwoLayerNetwork(
        input_dim=2,
        hidden_dim=5,   # Need 5 breaklines for inversion gadget
        output_dim=2
    )

    trainer = Trainer(
        network=network,
        learning_rate=0.01,
        max_iterations=5000,
        tolerance=1e-4,
        verbose=True
    )

    history = trainer.train(dataset)

    # Visualize results
    print("\n5. Visualizing results...")
    NetworkVisualizer.plot_training_history(
        history,
        save_path=config.plots_dir / "inversion_gadget_training.png"
    )

    NetworkVisualizer.plot_network_predictions(
        network,
        dataset,
        save_path=config.plots_dir / "inversion_gadget_predictions.png"
    )

    # Evaluate
    results = trainer.evaluate(dataset)
    print("\n6. Evaluation Results:")
    print(f"   MSE Loss: {results['mse']:.6e}")
    print(f"   Max Error: {results['max_error']:.6e}")
    print(f"   Is Exact Fit: {results['is_exact_fit']}")

    return gadget, network, history


def experiment_3_addition_constraint():
    """
    Experiment 3: Test Addition Constraint

    Purpose:
    - Test linear constraint X + Y = Z
    - Demonstrate constraint encoding with measuring lines
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Addition Constraint Test")
    print("=" * 60)

    config = get_config()

    # Create 3 variable gadgets
    print("\n1. Creating variable gadgets...")
    X, Y, Z = 0.7, 0.8, 1.5
    print(f"    X = {X:.3f}, Y = {Y:.3f}, Z = {Z:.3f}")
    print(f"    Constraint: X + Y = Z => {X:.3f} + {Y:.3f} = {Z:.3f}")
    print(f"    Expected Z = {X+Y:.3f}, Actual Z = {Z:.3f}")

    gadget_X = VariableGadget(value=X, orientation_angle=0.0, position=(0, 0))
    gadget_Y = VariableGadget(value=Y, orientation_angle=0.0, position=(0, 15))
    gadget_Z = VariableGadget(value=Z, orientation_angle=0.0, position=(0, 30))

    # Generate data points
    print("\n2. Generating data points...")
    all_points = []

    dataset_X = gadget_X.generate_data_points()
    dataset_Y = gadget_Y.generate_data_points()
    dataset_Z = gadget_Z.generate_data_points()

    all_points.extend(dataset_X.points)
    all_points.extend(dataset_Y.points)
    all_points.extend(dataset_Z.points)

    # Add constraint point
    # Intersection of upper measuring lines of X, Y and lower measuring line of Z
    intersection = np.array([0.0, 15.0])    # Simplified
    constraint_point = VariableConstraintEncoder.create_addition_constraint(
        gadget_X, gadget_Y, gadget_Z, intersection
    )
    all_points.append(constraint_point)

    dataset = Dataset(all_points)
    print(f'    Generated {len(dataset)} total data points')

    # Train network
    print("\n3. Training network...")
    network = TwoLayerNetwork(
        input_dim=2,
        hidden_dim=12,  # 3 gadgets x 4 breaklines each
        output_dim=2
    )

    trainer = Trainer(
        network=network,
        learning_rate=0.005,
        max_iterations=10000,
        tolerance=1e-4,
        verbose=True
    )

    history = trainer.train(dataset)

    # Visualize
    print("\n4. Visualizing results...")
    NetworkVisualizer.plot_training_history(
        history,
        save_path=config.plots_dir / "addition_constraint_training.png"
    )

    # Evaluate
    results = trainer.evaluate(dataset)
    print("\n5. Evaluation Result:")
    print(f"   MSE Loss: {results['mse']:.6e}")
    print(f"   Max Error: {results['max_error']:.6e}")
    print(f"   Is Exact Fit: {results['is_exact_fit']}")

    return network, history


def experiment_4_full_reduction():
    """
    Experiment 4: Test Full Reduction from ETR-Inv

    Purpose:
    - Demonstrate full reduction pipeline
    - Test with simple ETR-Inv instance
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Full Reduction Test")
    print("=" * 60)

    config = get_config()

    # Create simple ETR-Inv problem: X·Y = 1
    print("\n1. Creating ETR-Inv problem...")
    problem = create_simple_problem()
    print(f"    {problem}")

    # Find solution
    print("\n2. Finding solution...")
    solution = {'X': 1.5, 'Y': 2.0/3.0}
    print(f"    Solution: {solution}")
    print(f"    Satisfiable: {problem.is_satisfiable(solution)}")

    # Reduce to Train-F2NN
    print("\n3. Reducing to Train-F2NN...")
    reducer = Reducer()
    nn_problem = reducer.reduce(problem, solution)

    stats = reducer.get_reduction_stats(problem)
    print(f"    Reduction Statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print(f"\n  Dataset size: {len(nn_problem.dataset)}")
    print(f"  Input dim: {nn_problem.input_dim}")
    print(f"  Output dim: {nn_problem.output_dim}")

    # Train network
    print("\n4. Training network on reduced problem...")
    network = TwoLayerNetwork(
        input_dim=nn_problem.input_dim,
        hidden_dim=stats['min_hidden_neurons'],
        output_dim=nn_problem.output_dim
    )

    trainer = Trainer(
        network=network,
        learning_rate=0.005,
        max_iterations=10000,
        tolerance=1e-4,
        verbose=True
    )

    history = trainer.train(nn_problem.dataset)

    # Visualize
    print("\n5. Visualizing results...")
    NetworkVisualizer.plot_training_history(
        history,
        save_path=config.plots_dir / "full_reduction_training.png"
    )

    # Verify reduction
    print("\n6. Verifying reduction...")
    is_valid = reducer.verify_reduction(problem, nn_problem, network)
    print(f"    Reduction valid: {is_valid}")

    return problem, nn_problem, network, history


def run_all_experiments():
    """Run all experiments"""
    print("=" * 60)
    print("Running All Experiments")
    print("Neural Network Training Complexity")
    print("Implementation of 'Training Fully Connected Neural Network is ∃R-Complete'")
    print("=" * 60)

    # Experiment 1
    gadget1, net1, hist1 = experiment_1_variable_gadget()

    # Experiment 2
    gadget2, net2, hist2 = experiment_2_inversion_gadget()

    # Experiment 3
    net3, hist3 = experiment_3_addition_constraint()

    # Experiment 4
    prob4, nn_prob4, net4, hist4 = experiment_4_full_reduction()

    return {
        'experiment_1': (gadget1, net1, hist1),
        'experiment_2': (gadget2, net2, hist2),
        'experiment_3': (net3, hist3),
        'experiment_4': (prob4, nn_prob4, net4, hist4),
    }


if __name__=="__main__":
    # Set up configuration
    config = ProjectConfig()

    # Run experiments
    results = run_all_experiments()

    print("\nExperiment suite finished successfully!")