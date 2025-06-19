from SingleLayerFwdProp import single_layer_forward_propagation

def full_forward_propagation(X, params_values, nn):

    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn): 
        layer_idx = idx + 1
        A_prev    = A_curr

        activationFunction = nn[layer]["activation"]
        W_curr             = params_values["W" + str(layer_idx)]
        b_curr             = params_values["b" + str(layer_idx)]
        A_curr, Z_curr     = single_layer_forward_propagation(A_prev, W_curr, b_curr, activationFunction)
        
        memory["A" + str(idx)]       = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    memory["A" + str(layer_idx)] = A_curr

    return A_curr, memory