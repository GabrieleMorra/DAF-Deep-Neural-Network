import numpy as np
from ActivationFunctions import *

# Create singleton activation function instance
_activation_functions = ActivationFunctions()

def GD_backward_propagation(dLdA_curr, Z_curr, A_prev, activation):

    m = A_prev.shape[0]
    # Use singleton instance instead of creating new object
    af = _activation_functions
    if activation == "relu":
        dLdZ_curr = af.Derivative_ReLU(dLdA_curr, Z_curr)
    elif activation == "sigmoid":
        dLdZ_curr = af.Derivative_Sigmoid(dLdA_curr, Z_curr)
    elif activation == "softmax":
        dLdZ_curr = af.Derivative_Softmax(dLdA_curr, Z_curr)
    elif activation == "tanh":
        dLdZ_curr = af.Derivative_Tanh(dLdA_curr, Z_curr)
    elif activation == "softplus":
        dLdZ_curr = af.Derivative_SoftPlus(dLdA_curr, Z_curr)
    elif activation == "elu":
        dLdZ_curr = af.Derivative_Elu(dLdA_curr, Z_curr)
    elif activation == "leaky_relu":
        dLdZ_curr = af.Derivative_Leaky_ReLU(dLdA_curr, Z_curr)
    else:
        raise Exception('Non-supported activation function')

    dW_curr = np.dot(A_prev.T, dLdZ_curr)/m
    db_curr = np.mean(dLdZ_curr, axis=0, keepdims=True)

    return dLdZ_curr, dW_curr, db_curr

