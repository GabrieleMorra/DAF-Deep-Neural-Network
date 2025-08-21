import numpy as np
from GetLossValue import get_loss_derivative
import SingleLayerBackwardProp as slbp

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

        dLdZ_curr, dW_curr, db_curr = slbp.GD_backward_propagation(dLdA_curr, Z_curr, A_prev, activationFunction)

        grads_values[dW_key] = dW_curr
        grads_values[db_key] = db_curr

        dLdA_curr = np.dot(dLdZ_curr, W_curr.T)          

    return grads_values