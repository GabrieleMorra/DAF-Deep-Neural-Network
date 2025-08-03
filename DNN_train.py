from Train import TrainNeuralNetwork
from Visualization import visualize_NN_results
from GetDatabase import get_database

import json
import pickle

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

TrainNeuralNetwork(nnModel, Database)

# with open(nnModel["NeuralNetworkModel"]["OutputFileName"]+'.pkl', 'rb') as f:
#     storedData = pickle.load(f)
# visualize_NN_results(
#     storedData['X'], 
#     storedData['Y'],
#     storedData['params_values'], 
#     storedData['nn'], 
#     storedData['min_data'], 
#     storedData['max_data'], 
#     storedData['outputIndexEntry'],
#     storedData['loss_history'],
#     storedData['X_valid'],
#     storedData['Y_valid'],
#     )