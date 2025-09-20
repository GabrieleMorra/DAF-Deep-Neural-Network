from .layers import init_layers
from .forward_propagation import full_forward_propagation
from .loss_functions import get_mean_loss
from .backward_propagation import full_backward_propagation
from .optimizers import weights_update
from .metrics import get_accuracy_value
from ..models.onnx_export import save_onnx

import pickle
import numpy as np
from itertools import islice
import time
import os
import threading

def train_fidelity_r2(y_true, y_pred):
    """Calculate R² training fidelity score as percentage"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return max(0, r2 * 100)  # Prevent negative values

def validation_fidelity_r2(y_true, y_pred):
    """Calculate R² validation fidelity score as percentage"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return max(0, r2 * 100)  # Prevent negative values

def test(X_valid, Y_valid, params_values, nn, train_loss, validation_loss, momentum, i, min_data, max_data, outputIndexEntry, training_fidelity, ShowEvery=200, silent_mode=False):
   
    Y_hat, _                = full_forward_propagation(X_valid, params_values, nn)
    accuracy_per_variable   = get_accuracy_value(Y_hat, Y_valid, min_data, max_data, outputIndexEntry)

    # VECTORIZED: Calculate R² for all output variables simultaneously
    ss_res = np.sum((Y_valid - Y_hat) ** 2, axis=0)  # Sum per column (per variable)
    ss_tot = np.sum((Y_valid - np.mean(Y_valid, axis=0)) ** 2, axis=0)  # Sum per column
    r2_values = 1 - (ss_res / ss_tot)
    r2_per_variable = np.maximum(0, r2_values * 100).tolist()  # Converti a lista, evita valori negativi
    
    validation_fidelity     = validation_fidelity_r2(Y_valid, Y_hat)
    
    longest_string = max(
        len(f"Epoch:{i}"),
        len(f"R² scores: {', '.join(f'{acc:.2f}%' for acc in accuracy_per_variable)}"),
        len(f"Mean train loss: {train_loss:.2E}"),
        len(f"Mean test loss: {validation_loss:.2E}"),
        len(f"Training fidelity: {training_fidelity:.3f}%"),
        len(f"Validation fidelity: {validation_fidelity:.3f}%"),
        len(f"Momentum: {momentum:.2f}")
    )
    separator = "=" * (longest_string + 4)

    if i % ShowEvery == 0 and not silent_mode:
        print(separator)
        print("+ {:^{}} +".format(f"Epoch:{i}", longest_string))
        print("+ {:^{}} +".format(f"R² scores: {', '.join(f'{acc:.2f}%' for acc in accuracy_per_variable)}",longest_string))
        print("+ {:^{}} +".format(f"Mean train loss: {train_loss:.2E}", longest_string))
        print("+ {:^{}} +".format(f"Mean test loss: {validation_loss:.2E}", longest_string))
        print("+ {:^{}} +".format(f"Training fidelity: {training_fidelity:.3f}%", longest_string)) 
        print("+ {:^{}} +".format(f"Validation fidelity: {validation_fidelity:.3f}%", longest_string)) 
        print("+ {:^{}} +".format(f"Momentum: {momentum:.2f}", longest_string))
        print(separator + "\n")

    return accuracy_per_variable, validation_fidelity, r2_per_variable

def TrainNeuralNetwork(nn, database, model_id=None, data_queue=None, silent_mode=False, pause_check_func=None):

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
    
    # Extract only layer definitions, excluding NeuralNetworkModel and Visualizations
    network_layers = {k: v for k, v in nn.items() if "Layer" in k}

    params_values, previous_grads_values    = init_layers(network_layers, seed = 1)
    loss_history                            = []
    accuracy_history                        = []
    momentum                                = 0
    momentum_history                        = []

    m = {p: 0 for p in params_values.keys()}
    v = {p: 0 for p in params_values.keys()}
    t = 0

    # Track training execution time
    start_time = time.time()

    # Send initial training start signal to GUI
    if data_queue is not None and model_id is not None:
        # Send initial signal to mark model as starting training (epoch 1 to ensure epoch > 0)
        data_queue.put((model_id, 1, [0.0] * len(outputEntryIndices), 0.0, 0.0, float('inf')))

    for i in range(epochs + 1):
        # Check for pause/deletion at same frequency as other operations (ShowEvery)
        if pause_check_func and i % ShowEvery == 0:  # Check every ShowEvery epochs
            check_result = pause_check_func()
            if check_result is True:  # Deleted
                print(f"[TERMINATED] {model_id} deleted at epoch {i}")
                return None
            elif check_result is False:  # Paused
                # Wait for resume (no spam messages)
                while pause_check_func() is False:
                    time.sleep(1)  # Check every second
                # Check again if deleted after resume
                if pause_check_func() is True:
                    print(f"[TERMINATED] {model_id} deleted during pause at epoch {i}")
                    return None
        
        Y_hat, memory                                     = full_forward_propagation(X, params_values, network_layers)
        train_loss                                        = get_mean_loss(Y_hat, Y, nn)
        
        grads_values                                      = full_backward_propagation(Y_hat, Y, memory, params_values, nn, network_layers)
        params_values, previous_grads_values, _, m, v, t  = weights_update(params_values, grads_values, nn, learning_rate, previous_grads_values, momentum, m, v, t)

        training_fidelity  = train_fidelity_r2(Y, Y_hat)
        
        # Calculate validation loss and metrics
        if i % ShowEvery == 0 or i == epochs:  # Calculate according to config
            # Calculate validation loss
            Y_hat_valid, _ = full_forward_propagation(X_valid, params_values, network_layers)
            validation_loss = get_mean_loss(Y_hat_valid, Y_valid, nn)
            
            # Only print to console every ShowEvery epochs (preserve original behavior)
            should_print = (i % ShowEvery == 0) and not silent_mode
            accuracy_per_variable, validation_fidelity, r2_per_variable = test(X_valid, Y_valid, params_values, network_layers, train_loss, validation_loss, momentum, i, min_data, max_data, outputEntryIndices, training_fidelity, ShowEvery if should_print else 999999, silent_mode)
        else:
            # Keep last values for plotting consistency
            if 'validation_fidelity' not in locals():
                validation_fidelity = 0
                validation_loss = train_loss  # fallback
                r2_per_variable = [0] * len(outputEntryIndices)
                accuracy_per_variable = [0] * len(outputEntryIndices)

        momentum_history.append(momentum)
        loss_history.append(train_loss)
        
        # Send data for real-time plotting if available
        if data_queue is not None and model_id is not None:
            if i % ShowEvery == 0:
                # Send comprehensive training data: model_id, epoch, r2_per_variable, training_fidelity, validation_fidelity, train_loss
                data_queue.put((model_id, i, r2_per_variable, training_fidelity, validation_fidelity, train_loss))
        
    # Calculate training execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # Save training data and model artifacts only for single model training (not sweep)
    session_dir = None
    if model_id is None:  # Single model training
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

        # Create timestamped output directory for this training session
        from datetime import datetime
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        session_dir = f"data/output/TrainedModel_{timestamp}"
        
        # Ensure session directory exists
        os.makedirs(session_dir, exist_ok=True)
        
        # Extract dataset name from InputFileName
        input_filename = nn["NeuralNetworkModel"]["InputFileName"]
        dataset_name = os.path.splitext(input_filename)[0]
        output_filename = f"{session_dir}/Trained_DNN_{dataset_name}"
        
        # Save the data in a pickle file for visualization and analysis
        with open(f'{output_filename}.pkl', 'wb') as f:
            pickle.dump(data, f)

        # Save the data in an onnx file
        output_path = f'{output_filename}.onnx'
        save_onnx(nn, database, params_values, network_layers, (1, len(inputEntryIndices)), output_path)

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

    # Display training results summary
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
        f"Mean train loss        : {train_loss:.2E}\n"
        f"Mean test loss         : {validation_loss:.2E}\n"
        f"Elapsed time           : {elapsed_time}\n"
        f"R² score per variable  :\n" + "\n".join(f"\t{database['headers'][i]}: {acc:.2f}%" for i, acc in zip(outputEntryIndices, accuracy_per_variable)) + "\n"
        "\n"
        "Training process completed"
        )

    # Send final completion signal to GUI
    if data_queue is not None and model_id is not None:
        # Send final results to GUI
        final_results = {
            'training_fidelity': training_fidelity,
            'validation_fidelity': validation_fidelity,
            'mean_loss': train_loss
        }
        data_queue.put((model_id, 'COMPLETED', final_results))

    return {
        'training_fidelity': training_fidelity,
        'validation_fidelity': validation_fidelity,
        'train_loss': train_loss,
        'validation_loss': validation_loss,
        'elapsed_time': elapsed_time,
        'accuracy_per_variable': accuracy_per_variable,
        'r2_per_variable': r2_per_variable,
        'params_values': params_values,
        'loss_history': loss_history,
        'session_dir': session_dir  # Add session directory path
    }