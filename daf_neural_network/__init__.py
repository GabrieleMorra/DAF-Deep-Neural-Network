"""
DAF Deep Neural Network Framework
A High-Performance Neural Network Training System for Scientific Computing and Engineering Applications
"""

__version__ = "1.0.0"
__author__ = "DAF Development Team"
__description__ = "A High-Performance Neural Network Training System for Scientific Computing and Engineering Applications"

# Core imports for easy access
from .core.trainer import TrainNeuralNetwork
from .data.preprocessing import get_database
from .visualization.scientific_plots import visualize_NN_results

__all__ = [
    "TrainNeuralNetwork",
    "get_database", 
    "visualize_NN_results"
]