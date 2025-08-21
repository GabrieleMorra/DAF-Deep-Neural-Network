import numpy as np

def get_accuracy_value(Y_hat, Y, min_data, max_data, outputIndexEntry, threshold=3): 

    # Denormalize the data to avoid dividing by zero
    # Y_hat = Y_hat * (max_data[outputIndexEntry] - min_data[outputIndexEntry]) + min_data[outputIndexEntry]
    # Y = Y * (max_data[outputIndexEntry] - min_data[outputIndexEntry]) + min_data[outputIndexEntry]

    # diff_percent = np.where(Y != 0, (Y_hat - Y) / np.abs(Y) * 100, 0)
    # cond = np.abs(diff_percent) < threshold
    # accuracy = np.mean(cond, axis=0) * 100

    # Calculate R² for each output variable
    num_outputs = Y.shape[1]  
    fidelity_scores = []
    for i in range(num_outputs):
        ss_res = np.sum((Y[:, i] - Y_hat[:, i]) ** 2)
        ss_tot = np.sum((Y[:, i] - np.mean(Y[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0  # Avoid division by zero
        r2 = max(0, r2) 
        fidelity_scores.append(r2*100)

    # Validation fidelity basata su errore normalizzato
    # ones = np.ones(Y.shape[1])
    # fidelity_scores = np.max(0, ( ones - np.mean(np.sqrt((Y - Y_hat)**2)/2, axis=0) ) * 100)

    return np.array(fidelity_scores)
