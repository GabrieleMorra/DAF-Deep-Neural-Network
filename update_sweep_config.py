import json

def update_sweep_configuration():
    """Update sweep configuration with dynamic neuron counts based on input size"""
    
    # Load the current configuration
    with open("NeuralNetworkSweep.json", "r") as f:
        config = json.load(f)
    
    # Get number of input features
    num_inputs = len(config["NeuralNetworkModel"]["inputEntryIndices"])
    
    print(f"Detected {num_inputs} input features")
    
    # Base neuron ranges (standard architectures)
    base_range = [5, 10, 15, 20, 30, 50, 100, 150, 200]
    
    # Calculate input-based neurons with intelligent spacing
    input_size = num_inputs
    double_input = 2 * num_inputs
    
    # Add input-based neurons only if they don't conflict with base range
    dynamic_neurons = []
    
    # Add input_size if it's not too close to existing values
    if not any(abs(input_size - base) <= 2 for base in base_range):
        dynamic_neurons.append(input_size)
        print(f"Added {input_size} neurons (1x input size)")
    else:
        print(f"Skipped {input_size} neurons (too close to existing values)")
    
    # Add double_input if it's not too close to existing values
    if not any(abs(double_input - base) <= 2 for base in base_range + dynamic_neurons):
        dynamic_neurons.append(double_input)
        print(f"Added {double_input} neurons (2x input size)")
    else:
        print(f"Skipped {double_input} neurons (too close to existing values)")
    
    # For larger input sizes, consider half-input size too
    if num_inputs >= 20:
        half_input = num_inputs // 2
        if not any(abs(half_input - base) <= 2 for base in base_range + dynamic_neurons):
            dynamic_neurons.append(half_input)
            print(f"Added {half_input} neurons (0.5x input size)")
    
    # Combine and sort
    first_hidden_range = sorted(list(set(base_range + dynamic_neurons)))
    
    # For second hidden layer, use a more conservative range
    second_base_range = [8, 10, 16, 20, 24, 30]
    second_dynamic = []
    
    # Add input-based neurons to second layer only if reasonable
    if input_size <= 50 and not any(abs(input_size - base) <= 2 for base in second_base_range):
        second_dynamic.append(input_size)
    
    if double_input <= 60 and not any(abs(double_input - base) <= 2 for base in second_base_range + second_dynamic):
        second_dynamic.append(double_input)
    
    second_hidden_range = sorted(list(set(second_base_range + second_dynamic)))
    
    # Update configuration
    config["SweepConfiguration"]["first_hidden_neurons_range"] = first_hidden_range
    config["SweepConfiguration"]["second_hidden_neurons_range"] = second_hidden_range
    
    # Save updated configuration
    with open("NeuralNetworkSweep.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"\nFinal ranges:")
    print(f"First hidden layer: {first_hidden_range}")
    print(f"Second hidden layer: {second_hidden_range}")
    print(f"Dynamic neurons added: {dynamic_neurons}")
    print("Configuration updated successfully!")

if __name__ == "__main__":
    update_sweep_configuration()