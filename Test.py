from FullFwdPropagation import full_forward_propagation
from GetAccuracyValue import get_accuracy_value
from ActivationFunctions import *

import numpy as np

def validation_fidelity_r2(y_true, y_pred):
    """Calculate R² validation fidelity score as percentage"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return max(0, r2 * 100)  # Prevent negative values

def test(X_valid, Y_valid, params_values, nn, mean_loss, momentum, i, min_data, max_data, outputIndexEntry, training_fidelity, ShowEvery=200, silent_mode=False):
   
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
        len(f"Mean loss function: {mean_loss:.2E}"),
        len(f"Training fidelity: {training_fidelity:.3f}%"),
        len(f"Validation fidelity: {validation_fidelity:.3f}%"),
        len(f"Momentum: {momentum:.2f}")
    )
    separator = "=" * (longest_string + 4)

    if i % ShowEvery == 0 and not silent_mode:
        print(separator)
        print("+ {:^{}} +".format(f"Epoch:{i}", longest_string))
        print("+ {:^{}} +".format(f"R² scores: {', '.join(f'{acc:.2f}%' for acc in accuracy_per_variable)}",longest_string))
        print("+ {:^{}} +".format(f"Mean loss function: {mean_loss:.2E}", longest_string))
        print("+ {:^{}} +".format(f"Training fidelity: {training_fidelity:.3f}%", longest_string)) 
        print("+ {:^{}} +".format(f"Validation fidelity: {validation_fidelity:.3f}%", longest_string)) 
        print("+ {:^{}} +".format(f"Momentum: {momentum:.2f}", longest_string))
        print(separator + "\n")

    return accuracy_per_variable, validation_fidelity, r2_per_variable