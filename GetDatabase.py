import pandas as pd
import numpy as np

def get_database(nn):
    # Loading the data
    data       = pd.read_csv(nn["NeuralNetworkModel"]["InputFileName"], delimiter=";")
    headers    = data.columns
    data       = np.array(data)
    
    # Mix data to avoid bias
    m, n       = data.shape

    # if data has non-numeric columns, convert them to numeric random between 0 and 1
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = np.random.rand(m)

    max_data   = np.amax(data, axis=0)
    min_data   = np.amin(data, axis=0)

    # Normalising input and splitting data into training and validation sets
    data = (data - min_data) / (max_data - min_data)
    np.random.shuffle(data)
    id_train = np.round(nn["NeuralNetworkModel"]["training_testing_ratio"]*m).astype(int)

    data_train = data[0:id_train, :]
    data_valid = data[id_train:m, :]

    outputEntryIndices = nn["NeuralNetworkModel"]["outputEntryIndices"]
    inputEntryIndices = nn["NeuralNetworkModel"]["inputEntryIndices"]

    X_train = data_train[:, inputEntryIndices]
    Y_train = data_train[:, outputEntryIndices]
    X_valid = data_valid[:, inputEntryIndices]
    Y_valid = data_valid[:, outputEntryIndices]

    Database = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_valid": X_valid,
        "Y_valid": Y_valid,
        "min_data": min_data,
        "max_data": max_data,
        "headers": headers
    }
    return Database