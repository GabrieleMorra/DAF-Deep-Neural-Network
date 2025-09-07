import numpy as np
from .activation_functions import ActivationFunctions

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

def full_forward_propagation(X, params_values, nn):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn): 
        layer_idx = idx + 1
        A_prev    = A_curr

        # Pre-generate string keys to avoid repeated concatenation
        W_key = f"W{layer_idx}"
        b_key = f"b{layer_idx}"
        A_key = f"A{idx}"
        Z_key = f"Z{layer_idx}"

        activationFunction = nn[layer]["activation"]
        W_curr             = params_values[W_key]
        b_curr             = params_values[b_key]
        A_curr, Z_curr     = single_layer_forward_propagation(A_prev, W_curr, b_curr, activationFunction)
        
        memory[A_key] = A_prev
        memory[Z_key] = Z_curr

    memory[f"A{layer_idx}"] = A_curr

    return A_curr, memory