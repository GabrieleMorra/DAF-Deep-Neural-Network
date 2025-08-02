from initLayers import init_layers
from FullFwdPropagation import full_forward_propagation
from GetLossValue import get_mean_loss
from FullBackwardProp import full_backward_propagation
from UpdateParams import weights_update
from Test import test
from save_onnx import save_onnx

import pickle
import numpy as np
from itertools import islice
import time

def train_fidelity_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return max(0, r2 * 100)  # Impedisce valori negativi


def TrainNeuralNetwork(nn, database, model_id=None, data_queue=None):

    X                   = database['X_train']
    Y                   = database['Y_train']
    X_valid             = database['X_valid']
    Y_valid             = database['Y_valid']
    epochs              = nn["NeuralNetworkModel"]["epochs"]
    learning_rate       = nn["NeuralNetworkModel"]["learning_rate"]
    min_data            = database['min_data']
    max_data            = database['max_data'] 
    outputEntryIndices  = nn["NeuralNetworkModel"]['outputEntryIndices']
    inputEntryIndices   = nn["NeuralNetworkModel"]['inputEntryIndices']
    headers             = database['headers']
    ShowEvery           = nn["NeuralNetworkModel"]['ShowTestEvery']
    silent_mode         = nn["NeuralNetworkModel"].get('silent_mode', False)
    
    network_layers = dict(islice(nn.items(), 1, None))

    params_values, previous_grads_values    = init_layers(network_layers, seed = 1)
    loss_history                            = []
    accuracy_history                        = []
    momentum                                = 0
    momentum_history                        = []

    m = {p: 0 for p in params_values.keys()}
    v = {p: 0 for p in params_values.keys()}
    t = 0

    # Voglio contare il tempo di esecuzione del training
    start_time = time.time()

    for i in range(epochs + 1): 
        Y_hat, memory                                     = full_forward_propagation(X, params_values, network_layers)
        mean_loss                                         = get_mean_loss(Y_hat, Y, nn)
        grads_values                                      = full_backward_propagation(Y_hat, Y, memory, params_values, nn, network_layers)
        params_values, previous_grads_values, _, m, v, t  = weights_update(params_values, grads_values, nn, learning_rate, previous_grads_values, momentum, m, v, t)

        training_fidelity  = train_fidelity_r2(Y, Y_hat)
        accuracy_per_variable, validation_fidelity, r2_per_variable = test(X_valid, Y_valid, params_values, network_layers, mean_loss, momentum, i, min_data, max_data, outputEntryIndices, training_fidelity, ShowEvery, silent_mode)

        momentum_history.append(momentum)
        loss_history.append(mean_loss)
        
        # Send data for real-time plotting if available
        if data_queue is not None and model_id is not None and i % ShowEvery == 0:
            data_queue.put((model_id, i, r2_per_variable))
        
    # Calcolo il tempo di esecuzione del training
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # Saving the data
    data = {
        'nn': nn,
        'X': X,
        'Y': Y,
        'X_valid': X_valid,
        'Y_valid': Y_valid,
        'min_data': min_data,
        'max_data': max_data,
        'inputEntryIndices': inputEntryIndices,
        'outputIndexEntry': outputEntryIndices,
        'params_values': params_values,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history,
        'headers': headers
    }

    # Save the data in a pickle file
    # with open(nn["NeuralNetworkModel"]["OutputFileName"]+'.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    # Save the data in an onnx file
    save_onnx(nn, database, params_values, network_layers, (1, len(inputEntryIndices)), nn["NeuralNetworkModel"]["OutputFileName"]+'.onnx')

    # Final log message

    dim1 = max(
    len(f"Training method: "),
    len(f"Epochs: "),
    len(f"Learning rate: "),
    len(f"Loss function: "),
    len(f"Training/testing ratio: "),
    len(f"Number of hidden layers: "),
    len(f"Number of neurons: "),
    len(f"Activation functions: ")
    )

    # Scrivi su terminale tutti i risultati ottenuti:
    if not silent_mode:
        print("\nTraining completed successfully"
        "\n\n"
        "The neural network has been trained with the following parameters:\n"
        f"{'Training method:':<{dim1}} {nn['NeuralNetworkModel']['UpdateMethod']}\n"
        f"{'Epochs:':<{dim1}} {epochs}\n"
        f"{'Learning rate:':<{dim1}} {learning_rate:.2E}\n"
        f"{'Loss function:':<{dim1}} {nn['NeuralNetworkModel']['loss']}\n"
        f"{'Training/testing ratio:':<{dim1}} {nn['NeuralNetworkModel']['training_testing_ratio']}\n"
        f"{'Number of hidden layers:':<{dim1}} {len(network_layers)-1}\n"
        f"{'Number of neurons:':<{dim1}} {', '.join(str(layer['output_dim']) for layer in list(network_layers.values())[:-1])}\n"
        f"{'Activation functions:':<{dim1}} {', '.join(layer['activation'] for layer in network_layers.values())}\n"
        "\n"
        "The following results have been obtained:\n"
        f"Training fidelity      : {training_fidelity:.3f}%\n"
        f"Validation fidelity    : {validation_fidelity:.3f}%\n"
        f"Mean loss function     : {mean_loss:.2E}\n"
        f"Elapsed time           : {elapsed_time}\n"
        f"R² score per variabile :\n" + "\n".join(f"\t{header}: {acc:.2f}%" for header, acc in zip(database['headers'][outputEntryIndices], accuracy_per_variable)) + "\n"
        "\n"
        "End of the program"
        )

    return {
        'training_fidelity': training_fidelity,
        'validation_fidelity': validation_fidelity,
        'mean_loss': mean_loss,
        'elapsed_time': elapsed_time,
        'accuracy_per_variable': accuracy_per_variable,
        'r2_per_variable': r2_per_variable
    } 