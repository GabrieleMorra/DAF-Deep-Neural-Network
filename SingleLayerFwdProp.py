import numpy as np
from ActivationFunctions import *

# Singleton activation function instance for memory efficiency
_activation_functions = ActivationFunctions()

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activationFunction):
    """
    Perform forward propagation for a single neural network layer.
    
    Args:
        A_prev: Activations from previous layer (input)
        W_curr: Weight matrix for current layer
        b_curr: Bias vector for current layer  
        activationFunction: Activation function name (string)
        
    Returns:
        tuple: (A_curr, Z_curr) - activated output and linear transformation
    """
    Z_curr = np.dot(A_prev, W_curr) + b_curr
    # Use singleton instance for performance optimization
    af = _activation_functions
    if activationFunction == "relu":
        A_curr = af.ReLU(Z_curr)
    elif activationFunction == "sigmoid":
        A_curr = af.Sigmoid(Z_curr)
    elif activationFunction == "softmax":
        A_curr = af.Softmax(Z_curr)
    elif activationFunction == "tanh":
        A_curr = af.Tanh(Z_curr)
    elif activationFunction == "softplus":
        A_curr = af.SoftPlus(Z_curr)
    elif activationFunction == "elu":
        A_curr = af.Elu(Z_curr)
    elif activationFunction == "leaky_relu":
        A_curr = af.Leaky_ReLU(Z_curr)
    else: 
        raise Exception(f" Non-supported activation function: {activationFunction} ")
        
    return A_curr, Z_curr