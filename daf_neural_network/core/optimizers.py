import numpy as np

def weights_update(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum, m, v, t):

    Method = nn["NeuralNetworkModel"]["UpdateMethod"]

    if Method == "GradientDescent":
        return UpdateMethod.gradient_descent(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum)
    elif Method == "GaussNewton":
        return UpdateMethod.Gauss_Newton(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum)
    elif Method == "Adam":
        return UpdateMethod.Adam(params_value, grads_values, m, v, t, learning_rate)
    elif Method == "LevenbergMarquardt":
        return UpdateMethod.Levenberg_Marquardt(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum)
    else:
        raise Exception(f" Non-supported update method: {Method} ")


class UpdateMethod(object):
    
    def gradient_descent(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum):
        
        # mean_grad = np.mean(np.concatenate([grads_values["dW1"].ravel(), grads_values["db1"].ravel(), grads_values["dW2"].ravel(), grads_values["db2"].ravel()]))
        # momentum = np.abs(5e-5 / mean_grad)
        # momentum = np.min([2, momentum])
        momentum = 0.0

        for layer_idx, layer in enumerate(nn, start=1):
            params_value["W" + str(layer_idx)] -= learning_rate * (grads_values["dW" + str(layer_idx)] + momentum * previous_grads_values["dW" + str(layer_idx)])
            params_value["b" + str(layer_idx)] -= learning_rate * (grads_values["db" + str(layer_idx)] + momentum * previous_grads_values["db" + str(layer_idx)])

        previous_grads_values = grads_values.copy()

        return params_value, previous_grads_values, momentum, 1, 1, 1
    

    def Adam(params_value, grads_values, m, v, t, learning_rate):

        beta1=0.9
        beta2=0.999
        epsilon=1e-8
        t += 1 
        
        for p in params_value.keys():
            # Aggiorna i momenti
            m[p] = beta1 * m.get(p, 0) + (1 - beta1) * grads_values["d" + p]
            v[p] = beta2 * v.get(p, 0) + (1 - beta2) * (grads_values["d" + p] ** 2)

            # Calcola i correttivi bias-corrected dei momenti
            m_corrected = m[p] / (1 - beta1 ** t)
            v_corrected = v[p] / (1 - beta2 ** t)

            # Aggiorna i parametri
            params_value[p] -= learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
    
        return params_value, 1, 1, m, v, t
    
    
    def Gauss_Newton(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum):
        
        pass
    
    
    def Levenberg_Marquardt(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum):
        
        # Initialize damping and workspace arrays only once
        if not hasattr(UpdateMethod.Levenberg_Marquardt, 'damping'):
            UpdateMethod.Levenberg_Marquardt.damping = 1e-3
            UpdateMethod.Levenberg_Marquardt.workspace = None
            UpdateMethod.Levenberg_Marquardt.hessian_buffer = None
            UpdateMethod.Levenberg_Marquardt.use_full_hessian = True
        
        # Extract network dimensions
        num_layers = len(nn) - 1
        
        # Initialize workspace on first call for memory efficiency
        if UpdateMethod.Levenberg_Marquardt.workspace is None:
            total_params = sum(np.prod(params_value[f"W{i}"].shape) + np.prod(params_value[f"b{i}"].shape) 
                             for i in range(1, num_layers + 1))
            
            # Pre-allocate all working arrays
            UpdateMethod.Levenberg_Marquardt.workspace = {
                'grad_vector': np.empty(total_params, dtype=np.float64),
                'delta_vector': np.empty(total_params, dtype=np.float64),
                'param_slices': {},
                'total_params': total_params
            }
            
            # Pre-compute parameter slices
            idx = 0
            for layer_idx in range(1, num_layers + 1):
                W_key = f"W{layer_idx}"
                b_key = f"b{layer_idx}"
                W_size = np.prod(params_value[W_key].shape)
                b_size = np.prod(params_value[b_key].shape)
                
                UpdateMethod.Levenberg_Marquardt.workspace['param_slices'][W_key] = (idx, idx + W_size, params_value[W_key].shape)
                idx += W_size
                UpdateMethod.Levenberg_Marquardt.workspace['param_slices'][b_key] = (idx, idx + b_size, params_value[b_key].shape)
                idx += b_size
            
            # Allocate Hessian buffer only if needed for small networks
            if total_params <= 500:  # Only for small networks
                UpdateMethod.Levenberg_Marquardt.hessian_buffer = np.empty((total_params, total_params), dtype=np.float64)
            else:
                UpdateMethod.Levenberg_Marquardt.use_full_hessian = False
        
        # Access pre-allocated workspace
        workspace = UpdateMethod.Levenberg_Marquardt.workspace
        grad_vector = workspace['grad_vector']
        delta_vector = workspace['delta_vector']
        param_slices = workspace['param_slices']
        total_params = workspace['total_params']
        damping = UpdateMethod.Levenberg_Marquardt.damping
        
        # Efficient in-place gradient flattening
        for layer_idx in range(1, num_layers + 1):
            W_key = f"W{layer_idx}"
            b_key = f"b{layer_idx}"
            
            W_start, W_end, W_shape = param_slices[W_key]
            b_start, b_end, b_shape = param_slices[b_key]
            
            grad_vector[W_start:W_end] = grads_values[f"d{W_key}"].ravel()
            grad_vector[b_start:b_end] = grads_values[f"d{b_key}"].ravel()
        
        # Choose algorithm based on network size and use full LM for better convergence
        if UpdateMethod.Levenberg_Marquardt.use_full_hessian and total_params <= 500:
            # Full Levenberg-Marquardt for small networks - but optimized
            hessian = UpdateMethod.Levenberg_Marquardt.hessian_buffer
            
            # Compute J^T*J efficiently using BLAS operations
            np.outer(grad_vector, grad_vector, out=hessian)
            
            # Add damping in-place
            np.fill_diagonal(hessian, hessian.diagonal() + damping)
            
            try:
                # Use Cholesky for positive definite matrices (faster than LU)
                L = np.linalg.cholesky(hessian)
                y = np.linalg.solve(L, -grad_vector)
                delta_vector[:] = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                # Fallback: add more damping and use LU
                np.fill_diagonal(hessian, hessian.diagonal() + damping * 10)
                try:
                    delta_vector[:] = np.linalg.solve(hessian, -grad_vector)
                except np.linalg.LinAlgError:
                    # Final fallback: diagonal approximation
                    np.divide(-grad_vector, grad_vector**2 + damping * 10 + 1e-8, out=delta_vector)
        else:
            # Block-diagonal approximation for large networks
            # Process each layer's Hessian block separately to reduce memory
            delta_idx = 0
            
            for layer_idx in range(1, num_layers + 1):
                # Process weights
                W_key = f"W{layer_idx}"
                W_start, W_end, W_shape = param_slices[W_key]
                W_grad = grad_vector[W_start:W_end]
                
                if len(W_grad) < 100:  # Small weight matrix - use full block
                    W_hess = np.outer(W_grad, W_grad) + damping * np.eye(len(W_grad))
                    try:
                        delta_vector[W_start:W_end] = np.linalg.solve(W_hess, -W_grad)
                    except np.linalg.LinAlgError:
                        delta_vector[W_start:W_end] = -W_grad / (W_grad**2 + damping + 1e-8)
                else:  # Large weight matrix - use diagonal
                    delta_vector[W_start:W_end] = -W_grad / (W_grad**2 + damping + 1e-8)
                
                # Process biases (always diagonal due to small size)
                b_key = f"b{layer_idx}"
                b_start, b_end, b_shape = param_slices[b_key]
                b_grad = grad_vector[b_start:b_end]
                delta_vector[b_start:b_end] = -b_grad / (b_grad**2 + damping + 1e-8)
        
        # Apply learning rate scaling in-place
        delta_vector *= learning_rate
        
        # Efficient parameter updates without reshape overhead
        for layer_idx in range(1, num_layers + 1):
            W_key = f"W{layer_idx}"
            b_key = f"b{layer_idx}"
            
            W_start, W_end, W_shape = param_slices[W_key]
            b_start, b_end, b_shape = param_slices[b_key]
            
            # In-place updates using views
            params_value[W_key] += delta_vector[W_start:W_end].reshape(W_shape)
            params_value[b_key] += delta_vector[b_start:b_end].reshape(b_shape)
        
        # Adaptive damping with better heuristics
        grad_norm = np.linalg.norm(grad_vector)
        if grad_norm > 1e-6:  # Decrease damping if making progress
            UpdateMethod.Levenberg_Marquardt.damping *= 0.9
        else:  # Increase damping if stagnating
            UpdateMethod.Levenberg_Marquardt.damping *= 1.1
        
        UpdateMethod.Levenberg_Marquardt.damping = np.clip(UpdateMethod.Levenberg_Marquardt.damping, 1e-8, 1e1)
        
        return params_value, previous_grads_values, momentum, 1, 1, 1