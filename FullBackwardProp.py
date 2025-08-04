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


    # Tentativo implementazione metodo di Gauss-Newton
    # e = (Y - Y_hat).reshape(-1, 1).astype(float)
    # g = np.vstack([
    #     grads_values["dW1"].reshape(-1, 1),
    #     grads_values["db1"].reshape(-1, 1),
    #     grads_values["dW2"].reshape(-1, 1),
    #     grads_values["db2"].reshape(-1, 1)
    # ])
    # Jacobian = np.dot(e, g.T).astype(np.float32)
    # # Risoluzione del sistema lineare anziché calcolare l'inversa
    # # Nota: np.linalg.solve risolve l'equazione ax = b. Qui, a = JtJ, b = np.dot(Jacobian.T, e)
    # JtJ = np.dot(Jacobian.T, Jacobian)
    # update = np.linalg.solve(JtJ, np.dot(Jacobian.T, e))
    # len_0, len_1, len_2 = grads_values["dW1"].size, grads_values["db1"].size, grads_values["dW2"].size
    # offsets = np.cumsum([0, len_0, len_1, len_2])
    # grads_values["dW1"] = update[offsets[0]:offsets[1]].reshape(grads_values["dW1"].shape).astype(float)
    # grads_values["db1"] = update[offsets[1]:offsets[2]].reshape(grads_values["db1"].shape).astype(float)
    # grads_values["dW2"] = update[offsets[2]:offsets[3]].reshape(grads_values["dW2"].shape).astype(float)
    # grads_values["db2"] = update[offsets[3]:].reshape(grads_values["db2"].shape).astype(float)

    return grads_values