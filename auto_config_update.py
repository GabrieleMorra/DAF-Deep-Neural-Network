import json
import os

def update_neural_network_configs():
    """Automatically update all neural network configuration files with dynamic values"""
    
    # Update NeuralNetwork.json
    if os.path.exists("NeuralNetwork.json"):
        with open("NeuralNetwork.json", "r") as f:
            config = json.load(f)
        
        num_inputs = len(config["NeuralNetworkModel"]["inputEntryIndices"])
        
        # Update hidden layer neurons to match input size
        if "FirstHiddenLayer" in config:
            config["FirstHiddenLayer"]["neurons"] = num_inputs
        if "SecondHiddenLayer" in config:
            config["SecondHiddenLayer"]["neurons"] = num_inputs
        
        with open("NeuralNetwork.json", "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"Updated NeuralNetwork.json with {num_inputs} neurons in hidden layers")
    
    # Update NeuralNetworkSweep.json
    if os.path.exists("NeuralNetworkSweep.json"):
        with open("NeuralNetworkSweep.json", "r") as f:
            config = json.load(f)
        
        num_inputs = len(config["NeuralNetworkModel"]["inputEntryIndices"])
        
        # Calculate dynamic neuron counts
        input_size = num_inputs
        double_input = 2 * num_inputs
        triple_input = 3 * num_inputs
        
        # Base neuron ranges
        base_range = [5, 10, 15, 50, 100, 150, 200]
        
        # Add input-size-based neurons
        dynamic_neurons = [input_size, double_input, triple_input]
        
        # Combine and sort
        first_hidden_range = sorted(list(set(base_range + dynamic_neurons)))
        second_hidden_range = sorted(list(set([10, 20, 30] + dynamic_neurons)))
        
        # Update configuration
        config["SweepConfiguration"]["first_hidden_neurons_range"] = first_hidden_range
        config["SweepConfiguration"]["second_hidden_neurons_range"] = second_hidden_range
        
        with open("NeuralNetworkSweep.json", "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"Updated NeuralNetworkSweep.json with dynamic ranges based on {num_inputs} inputs")
        print(f"  - Added values: {input_size}, {double_input}, {triple_input}")

if __name__ == "__main__":
    print("Auto-updating neural network configurations...")
    update_neural_network_configs()
    print("Configuration update completed!")