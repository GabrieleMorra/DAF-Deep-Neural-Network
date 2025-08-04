from SingleLayerFwdProp import single_layer_forward_propagation

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