from FullFwdPropagation import full_forward_propagation
from GetAccuracyValue import get_accuracy_value
from ActivationFunctions import *

import numpy as np

def validation_fidelity_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return max(0, r2 * 100)  # Impedisce valori negativi

def test(X_valid, Y_valid, params_values, nn, mean_loss, momentum, i, min_data, max_data, outputIndexEntry, training_fidelity, ShowEvery=200):
   
    Y_hat, _                = full_forward_propagation(X_valid, params_values, nn)
    accuracy_per_variable   = get_accuracy_value(Y_hat, Y_valid, min_data, max_data, outputIndexEntry)

    validation_fidelity           = ', '.join(f"{err:.2f}%" for err in accuracy_per_variable)
    validation_fidelity           = validation_fidelity_r2(Y_valid, Y_hat)
    
    longest_string = max(
        len(f"Epoch:{i}"),
        len(f"R² scores: {', '.join(f'{acc:.2f}%' for acc in accuracy_per_variable)}"),
        len(f"Mean loss function: {mean_loss:.2E}"),
        len(f"Training fidelity: {training_fidelity:.3f}%"),
        len(f"Validation fidelity: {validation_fidelity:.3f}%"),
        len(f"Momentum: {momentum:.2f}")
    )
    separator = "=" * (longest_string + 4)

    if i % ShowEvery == 0:
        print(separator)
        print("+ {:^{}} +".format(f"Epoch:{i}", longest_string))
        print("+ {:^{}} +".format(f"R² scores: {', '.join(f'{acc:.2f}%' for acc in accuracy_per_variable)}",longest_string))
        print("+ {:^{}} +".format(f"Mean loss function: {mean_loss:.2E}", longest_string))
        print("+ {:^{}} +".format(f"Training fidelity: {training_fidelity:.3f}%", longest_string)) 
        print("+ {:^{}} +".format(f"Validation fidelity: {validation_fidelity:.3f}%", longest_string)) 
        print("+ {:^{}} +".format(f"Momentum: {momentum:.2f}", longest_string))
        print(separator + "\n")

    return accuracy_per_variable, validation_fidelity