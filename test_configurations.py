import json
from itertools import product

# Load configuration
with open("NeuralNetworkSweep.json", "r") as f:
    sweep_config = json.load(f)

first_hidden_neurons_range = sweep_config["SweepConfiguration"]["first_hidden_neurons_range"]
second_hidden_neurons_range = sweep_config["SweepConfiguration"]["second_hidden_neurons_range"]
total_layers_range = sweep_config["SweepConfiguration"]["total_layers_range"]

# Generate all combinations
all_combinations = list(product(first_hidden_neurons_range, second_hidden_neurons_range, total_layers_range))

# Filter out duplicate architectures for 2-layer networks
unique_combinations = []
seen_architectures = set()

for first_hidden, second_hidden, total_layers in all_combinations:
    hidden_layers = total_layers - 1
    
    # Create architecture identifier
    if hidden_layers == 1:
        arch_id = f"{first_hidden}"  # Only first hidden layer matters for 1-layer networks
        # For 1-layer networks, use any second_hidden value (we'll use the first one we see)
        if arch_id not in seen_architectures:
            unique_combinations.append((first_hidden, second_hidden, total_layers))
            seen_architectures.add(arch_id)
    else:
        arch_id = f"{first_hidden}x{second_hidden}"  # Both layers matter for 2-layer networks
        if arch_id not in seen_architectures:
            unique_combinations.append((first_hidden, second_hidden, total_layers))
            seen_architectures.add(arch_id)

param_combinations = unique_combinations

print(f"Original combinations: {len(all_combinations)}")
print(f"Unique architectures after filtering: {len(param_combinations)}")
print(f"Eliminated {len(all_combinations) - len(param_combinations)} duplicate architectures")
print()

print("=== NEURAL NETWORK SWEEP CONFIGURATIONS ===")
print(f"Total configurations: {len(param_combinations)}")
print(f"First hidden neurons range: {first_hidden_neurons_range}")
print(f"Second hidden neurons range: {second_hidden_neurons_range}")
print(f"Total layers range: {total_layers_range}")
print()

print("All configurations to be tested:")
print("ID  | 1st H | 2nd H | Layers | Network Architecture")
print("----|-------|-------|--------|---------------------")

for i, (first_h, second_h, total_layers) in enumerate(param_combinations):
    model_id = f"{first_h}_{second_h}_{total_layers}L"
    hidden_layers = total_layers - 1
    
    # Architecture shows only hidden layers: first_hidden x second_hidden
    # InputLayer and OutputLayer are not shown, only hidden layers count
    if hidden_layers == 1:
        description = f"{first_h}"
    else:
        description = f"{first_h}x{second_h}"
    
    print(f"{i+1:3d} | {first_h:5d} | {second_h:5d} | {total_layers:6d} | {description}")

print(f"\nSpecial configurations with 13 and 26 neurons (input size and double input size):")
special_configs = [(first_h, second_h, total_layers) for first_h, second_h, total_layers in param_combinations 
                  if first_h in [13, 26] or second_h in [13, 26]]

for first_h, second_h, total_layers in special_configs:
    hidden_layers = total_layers - 1
    if hidden_layers == 1:
        arch = f"{first_h}"
    else:
        arch = f"{first_h}x{second_h}"
    print(f"  • {first_h}_{second_h}_{total_layers}L: {arch}")