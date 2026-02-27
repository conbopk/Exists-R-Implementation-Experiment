"""
Setup script for Neural Network Complexity Experiment
"""
from setuptools import setup, find_packages
from pathlib import Path


# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="neural-network-complexity",
    version="0.1.0",
    author="Implementation based on Bertschinger et al. paper",
    description="Implementation of 'Training Fully Connected Neural Networks is ∃R-Complete'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/conbopk/Exists-R-Implementation-Experiment.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.10.8",
        "numpy>=2.4.2",
        "pandas>=3.0.1",
        "pillow>=12.1.1",
        "PyYAML>=6.0.3",
        "seaborn>=0.13.2",
        "tqdm>=4.67.3",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "jupyter>=1.0.0",
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-experiments=src.experiments.run_experiments:run_all_experiments",
        ],
    },
)