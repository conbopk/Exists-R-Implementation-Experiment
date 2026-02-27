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
    url=""
)