"""
Configuration Management Module
"""
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class NetworkConfig:
    """Configuration for Neural Network architecture"""
    input_dim: int = 2
    output_dim: int = 2
    hidden_dim: int = 10
    activation: str = 'relu'


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    learning_rate: float = 0.001
    max_iterations: int = 10000
    tolerance: float = 1e-6
    batch_size: Optional[int] = None    # None = full batch
    optimizer: str = "adam"


@dataclass
class GadgetConfig:
    """Configuration for gadgets"""
    variable_slope_range: tuple = (1.5, 3.0)    # (3/2, 3) from paper
    inversion_num_breaklines: int = 5
    measuring_line_distance: float = 1.0


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    random_seed: int = 42
    num_trials: int = 10
    save_plots: bool = True
    save_results: bool = True
    verbose: bool = True


@dataclass
class ProjectConfig:
    """Main project configuration"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gadget: GadgetConfig = field(default_factory=GadgetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize paths after object creation"""
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.plots_dir = self.results_dir / "plots"
        self.logs_dir = self.results_dir / "logs"

        # Create directories if they don't exist
        for directory in [self.data_dir, self.results_dir, self.plots_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectConfig":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            network=NetworkConfig(**config_dict.get('network', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            gadget=GadgetConfig(**config_dict.get('gadget', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )

    def to_yaml(self, path: Path):
        """Save configuration to YAML file"""
        config_dict = {
            'network': {
                'input_dim': self.network.input_dim,
                'hidden_dim': self.network.hidden_dim,
                'output_dim': self.network.output_dim,
                'activation': self.network.activation,
            },
            'training': {
                'learning_rate': self.training.learning_rate,
                'max_iterations': self.training.max_iterations,
                'tolerance': self.training.tolerance,
                'batch_size': self.training.batch_size,
                'optimizer': self.training.optimizer,
            },
            'gadget': {
                'variable_slope_range': self.gadget.variable_slope_range,
                'inversion_num_breaklines': self.gadget.inversion_num_breaklines,
                'measuring_line_distance': self.gadget.measuring_line_distance,
            },
            'experiment': {
                'random_seed': self.experiment.random_seed,
                'num_trials': self.experiment.num_trials,
                'save_plots': self.experiment.save_plots,
                'save_results': self.experiment.save_results,
                'verbose': self.experiment.verbose,
            }
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'network': self.network.__dict__,
            'training': self.training.__dict__,
            'gadget': self.gadget.__dict__,
            'experiment': self.experiment.__dict__,
        }


# Global configuration instance
_global_config: Optional[ProjectConfig] = None


def get_config() -> ProjectConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = ProjectConfig()
    return _global_config


def set_config(config: ProjectConfig):
    """Set global configuration instance"""
    global _global_config
    _global_config = config


def load_config(path: Path) -> ProjectConfig:
    """Load and set global configuration from file"""
    config = ProjectConfig.from_yaml(path)
    set_config(config)
    return config