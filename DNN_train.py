from Train import TrainNeuralNetwork
from Visualization import visualize_NN_results
from GetDatabase import get_database

import json
import pickle
import os

def convert_json_format(new_nn):
    old_nn = {}
    keys = list(new_nn.keys())
    last_layer_key = next(key for key in keys[::-1] if "Layer" in key)

    previous_neurons = len(new_nn["NeuralNetworkModel"]["inputEntryIndices"])

    for key, value in new_nn.items():
        if "Layer" in key:
            if key == last_layer_key:
                # Gestisci l'ultimo layer separatamente
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
            # Copia la sezione "NeuralNetworkModel" senza modifiche
            old_nn[key] = value

    return old_nn

JSONFileName = "NeuralNetwork.json"

with open(JSONFileName, "r") as f:
    read_json = json.load(f)
nnModel = convert_json_format(read_json)

Database = get_database(nnModel)

result = None

# Extract dataset name for automatic naming
input_filename = nnModel["NeuralNetworkModel"]["InputFileName"]
dataset_name = os.path.splitext(input_filename)[0]
output_file = f"Trained_DNN_{dataset_name}"

# Train the neural network and get results
result = TrainNeuralNetwork(nnModel, Database)


if result is not None:
    print("\nGenerating comprehensive visualizations...")
    
    # Call visualization with PKL file
    from Visualization import visualize_NN_results
    visualize_NN_results()
else:
    try:
        with open(f"{output_file}.pkl", "rb") as f:
            result = pickle.load(f)
            print(f"✅ Successfully loaded trained model from '{output_file}.pkl'")

            from Visualization import visualize_NN_results
            visualize_NN_results()
    except FileNotFoundError:
        print(f"❌ '{output_file}.pkl' not found. Please ensure the model is trained and saved.")
    except Exception as e:
        print(f"❌ Error loading '{output_file}.pkl': {e}")