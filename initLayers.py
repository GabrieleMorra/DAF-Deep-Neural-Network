import numpy as np

# He Initialization:
# This technique initializes the weights from a normal distribution 
# with a mean of 0 and variance of 2/n_in, where n_in is the number 
# of input units to the layer. It is particularly effective for 
# layers using ReLU activation functions, as it helps maintain 
# the variance of the activations throughout the network, 
# preventing vanishing or exploding gradients during training.

def init_layers(nn, seed):
    np.random.seed(seed)
    params_values           = {}
    previous_grads_values   = {}

    for idx, layer in enumerate(nn):
        dic = nn[layer]
        # print(" === Initialization of Neural Network  === ")
        # print(" Layer: ", layer)
        layer_idx = idx + 1
        # print(" Content: ", dic)

        # Ottieni le dimensioni di input e output dal dizionario
        layer_input_size = dic["input_dim"]
        layer_output_size = dic["output_dim"]

        # Inizializzazione dei pesi e dei bias
        params_values['W' + str(layer_idx)] = np.random.randn(layer_input_size, layer_output_size) * np.sqrt(2 / layer_input_size)
        params_values['b' + str(layer_idx)] = np.random.randn(1, layer_output_size) * np.sqrt(2 / 1)

        # Inizializzazione dei gradienti a zero
        previous_grads_values['dW' + str(layer_idx)] = np.zeros((layer_input_size, layer_output_size))
        previous_grads_values['db' + str(layer_idx)] = np.zeros((1, layer_output_size))

    return params_values, previous_grads_values