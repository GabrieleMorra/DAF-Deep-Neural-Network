import numpy as np

def weights_update(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum, m, v, t):

    Method = nn["NeuralNetworkModel"]["UpdateMethod"]

    if Method == "GradientDescent":
        return UpdateMethod.gradient_descent(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum)
    elif Method == "GaussNewton":
        return UpdateMethod.Gauss_Newton(params_value, grads_values, nn, learning_rate, previous_grads_values, momentum)
    elif Method == "Adam":
        return UpdateMethod.Adam(params_value, grads_values, m, v, t, learning_rate)
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