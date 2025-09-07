import json
import os

def load_config(config_path):
    """Load configuration from JSON file"""
    possible_paths = [
        config_path,
        f"configs/{config_path}",
        os.path.join(os.getcwd(), config_path),
        os.path.join(os.getcwd(), "configs", config_path)
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError(f"Configuration file {config_path} not found in any expected location")

def convert_json_format(new_nn):
    """Convert new JSON format to legacy format for backward compatibility"""
    old_nn = {}
    keys = list(new_nn.keys())
    last_layer_key = next(key for key in keys[::-1] if "Layer" in key)

    previous_neurons = len(new_nn["NeuralNetworkModel"]["inputEntryIndices"])

    for key, value in new_nn.items():
        if "Layer" in key:
            if key == last_layer_key:
                # Handle output layer separately with appropriate dimensions
                old_nn[key] = {
                    "input_dim": previous_neurons,
                    "output_dim": len(new_nn["NeuralNetworkModel"]["outputEntryIndices"]),
                    "activation": value["activation"]
                }
            else:
                old_nn[key] = {
                    "input_dim": previous_neurons,
                    "output_dim": value["neurons"],
                    "activation": value["activation"]
                }
                previous_neurons = value["neurons"]
        else:
            # Copy "NeuralNetworkModel" section without modifications
            old_nn[key] = value

    return old_nn