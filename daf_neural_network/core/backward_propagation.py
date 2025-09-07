import numpy as np
from .loss_functions import get_loss_derivative
from .activation_functions import ActivationFunctions

# Create singleton activation function instance
_activation_functions = ActivationFunctions()

def single_layer_backward_propagation(dLdA_curr, Z_curr, A_prev, activation):
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

def full_backward_propagation(Y_hat, Y, memory, params_value, nn, network_layers):
    grads_values = {}
    Y            = Y.reshape(Y_hat.shape)

    dLdA_curr = get_loss_derivative(Y_hat, Y, nn)

    for layer_idx_prev, layer in reversed(list(enumerate(network_layers))):
        dic                = nn[layer]
        layer_idx_curr     = layer_idx_prev + 1
        activationFunction = dic["activation"]

        # Pre-generate string keys to avoid repeated concatenation
        A_key = f"A{layer_idx_prev}"
        Z_key = f"Z{layer_idx_curr}"
        W_key = f"W{layer_idx_curr}"
        dW_key = f"dW{layer_idx_curr}"
        db_key = f"db{layer_idx_curr}"

        A_prev = memory[A_key]
        Z_curr = memory[Z_key]
        W_curr = params_value[W_key]

        dLdZ_curr, dW_curr, db_curr = single_layer_backward_propagation(dLdA_curr, Z_curr, A_prev, activationFunction)

        grads_values[dW_key] = dW_curr
        grads_values[db_key] = db_curr

        dLdA_curr = np.dot(dLdZ_curr, W_curr.T)          

    return grads_values