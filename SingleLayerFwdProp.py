import numpy as np
from ActivationFunctions import *

# Create singleton activation function instance
_activation_functions = ActivationFunctions()

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activationFunction):
    
    Z_curr = np.dot(A_prev, W_curr) + b_curr
    # Use singleton instance instead of creating new object
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