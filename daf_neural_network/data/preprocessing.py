import pandas as pd
import numpy as np

def get_database(nn):
    """
    Load and preprocess dataset for neural network training.
    
    Args:
        nn: Neural network configuration dictionary
        
    Returns:
        Database: Dictionary containing training/validation data and metadata
    """
    # Load dataset from CSV file with proper path handling
    input_file = nn["NeuralNetworkModel"]["InputFileName"]
    # Try different possible paths for the input file
    possible_paths = [
        input_file,
        f"data/input/{input_file}",
        f"data/{input_file}",
        input_file  # fallback to original path
    ]
    
    data = None
    for path in possible_paths:
        try:
            data = pd.read_csv(path, delimiter=nn["NeuralNetworkModel"]["Delimiter"])
            break
        except FileNotFoundError:
            continue
    
    if data is None:
        raise FileNotFoundError(f"Could not find input file {input_file} in any expected location")
    
    headers    = list(data.columns)
    data       = np.array(data)
    
    # Check for NaN values and report their locations
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        nan_positions = np.where(nan_mask)
        print(f"Found {np.sum(nan_mask)} NaN values at positions:")
        for i in range(len(nan_positions[0])):
            row, col = nan_positions[0][i], nan_positions[1][i]
            print(f"  Row {row}, Column {col} ({headers[col]})")
    
    # Shuffle data to avoid bias in training
    m, n       = data.shape
            
    max_data   = np.amax(data, axis=0)
    min_data   = np.amin(data, axis=0)

    # Set seed for reproducible data shuffling
    np.random.seed(1)  # Set seed for reproducible shuffling
    np.random.shuffle(data)
    id_train = np.round(nn["NeuralNetworkModel"]["training_testing_ratio"]*m).astype(int)

    data_train = data[0:id_train, :]
    data_valid = data[id_train:m, :]

    outputEntryIndices = nn["NeuralNetworkModel"]["outputEntryIndices"]
    inputEntryIndices = nn["NeuralNetworkModel"]["inputEntryIndices"]

    # Extract raw data
    X_train_raw = data_train[:, inputEntryIndices]
    Y_train_raw = data_train[:, outputEntryIndices]
    X_valid_raw = data_valid[:, inputEntryIndices]
    Y_valid_raw = data_valid[:, outputEntryIndices]
    
    # Normalize all data using global min-max scaling for consistency
    X_train = (X_train_raw - min_data[inputEntryIndices]) / (max_data[inputEntryIndices] - min_data[inputEntryIndices])
    Y_train = (Y_train_raw - min_data[outputEntryIndices]) / (max_data[outputEntryIndices] - min_data[outputEntryIndices])
    X_valid = (X_valid_raw - min_data[inputEntryIndices]) / (max_data[inputEntryIndices] - min_data[inputEntryIndices])
    Y_valid = (Y_valid_raw - min_data[outputEntryIndices]) / (max_data[outputEntryIndices] - min_data[outputEntryIndices])

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