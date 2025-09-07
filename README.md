# DAF Deep Neural Network Framework

A high-performance neural network training system for systematic architecture exploration and optimization.

## Overview

The DAF Neural Network Framework provides automated architecture exploration with multi-threaded parallel training, real-time monitoring, and comprehensive model analysis. The system includes integrated SHAP analysis for model interpretability and exports ONNX-compliant models for deployment.

## Key Features

- **Automated Architecture Sweep**: Systematic exploration of network architectures
- **Real-time GUI Monitoring**: Live tracking of training progress and metrics  
- **Multi-threaded Training**: Parallel execution with persistent thread pools
- **Dynamic Configuration**: Add new architectures during training
- **Model Interpretability**: Integrated SHAP analysis for feature importance
- **ONNX Export**: Standardized model format for deployment
- **Scientific Visualization**: Publication-quality plots and analysis

## 1. Introduction

Deep neural networks have become indispensable tools in scientific computing, particularly for surrogate modeling, parameter estimation, and multi-physics simulations. However, the selection of optimal network architectures and understanding of model decision-making processes remain computationally intensive and often opaque challenges. This framework addresses these issues by providing:

1. **Systematic Architecture Exploration**: Automated sweep across predefined architecture spaces
2. **Parallel Computing Optimization**: True multi-core parallelization with CPU affinity management  
3. **Real-time Performance Analytics**: Live monitoring of training metrics and convergence behavior
4. **Model Interpretability**: Integrated SHAP analysis for explainable AI and feature importance ranking
5. **Scientific Visualization**: Publication-quality plots and analysis tools for research documentation
6. **Standardized Model Export**: ONNX-compliant models for research reproducibility and deployment

## Project Structure

```
daf_neural_network/
├── core/                       # Core training components
│   ├── trainer.py             # Main training engine
│   ├── layers.py              # Layer initialization
│   ├── forward_propagation.py # Forward pass implementation
│   ├── backward_propagation.py # Backward pass implementation
│   ├── optimizers.py          # Optimization algorithms
│   ├── loss_functions.py      # Loss function implementations
│   └── metrics.py             # Performance metrics
├── data/                      # Data handling
│   └── preprocessing.py       # Data preprocessing and normalization
├── gui/                       # Graphical interface
│   └── realtime_monitor.py    # Real-time training monitor
├── models/                    # Model export/import
│   └── onnx_export.py        # ONNX model serialization
├── utils/                     # Utilities
│   ├── config.py             # Configuration management
│   ├── helpers.py            # Helper functions
│   └── visualization.py      # Scientific visualization
├── run_architecture_sweep.py  # Multi-threaded sweep orchestrator
├── train_single_model.py     # Single model training
└── configs/                   # Configuration files
    └── architecture_sweep.json # Sweep parameters
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/DAF-Deep-Neural-Network.git
cd DAF-Deep-Neural-Network
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Architecture Sweep
Run automated architecture exploration with real-time GUI monitoring:

```bash
python run_architecture_sweep.py
```

### Single Model Training
Train a specific architecture:

```bash
python train_single_model.py
```

## Configuration

### Architecture Sweep Configuration (`configs/architecture_sweep.json`)

```json
{
  "SweepConfiguration": {
    "first_hidden_neurons_range": [5, 10, 20],
    "second_hidden_neurons_range": [5, 10],
    "total_layers_range": [1, 2],
    "max_threads": 4
  },
  "NeuralNetworkModel": {
    "loss": "LC",
    "UpdateMethod": "Adam",
    "learning_rate": 1e-3,
    "epochs": 10000,
    "inputEntryIndices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "outputEntryIndices": [13],
    "InputFileName": "Boston_Housing.csv",
    "ShowTestEvery": 500
  }
}
```

## GUI Features

The real-time monitoring interface provides:

- **Architecture Performance Table**: Live tracking of training metrics
- **Interactive Controls**: Pause/resume training, add new architectures  
- **Dynamic Scrolling**: Automatic table resizing with scroll support
- **Model Management**: Right-click context menus for model control
- **Export Capabilities**: CSV export of training results
- **Visual Progress**: Real-time R² scores and loss tracking

## Output Structure

### Single Model Training
```
data/output/TrainedModel_[DDMMYY_HHMMSS]/
├── Trained_DNN_[dataset].pkl          # Trained model with metadata
├── Trained_DNN_[dataset].onnx         # ONNX-exported model
└── visualizations/                    # Scientific plots and analysis
    ├── training_convergence.png       # Loss curves
    ├── training_convergence.pdf       # High-resolution version
    └── shap_analysis_*.png            # SHAP interpretability plots
```

### Architecture Sweep
```
data/output/
└── sweep_results.csv                  # Architecture comparison results
```

## Technical Details

### Multi-threading Architecture
- **Persistent Thread Pool**: Threads remain active throughout training
- **Dynamic Job Addition**: New architectures can be added during execution
- **Resource Management**: Configurable thread count with CPU affinity
- **Thread-safe Communication**: Queue-based inter-thread messaging

### Performance Optimizations
- **Vectorized Operations**: NumPy-optimized mathematical computations
- **Memory Management**: Efficient data copying and garbage collection
- **GUI Responsiveness**: Non-blocking interface updates
- **Conditional Scrollbars**: Dynamic UI element visibility

## Requirements

- Python 3.11+
- NumPy 1.24+
- pandas
- matplotlib
- scikit-learn
- onnx
- onnxruntime
- tkinter (usually included with Python)

## License

This project is licensed under the MIT License.