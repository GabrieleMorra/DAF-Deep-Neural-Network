import onnx
import numpy as np
from onnx import helper, TensorProto

def save_onnx(nn, database, params_values, network_layers, input_shape, output_file):
    """
    Save the neural network in ONNX file format with normalization/denormalization.

    :param nn: Neural network model dictionary
    :param database: Database containing min/max values for normalization
    :param params_values: Dictionary containing the parameters of the neural network.
    :param network_layers: Dictionary defining the structure of the neural network.
    :param input_shape: Tuple representing the shape of the input data.
    :param output_file: Name of the output ONNX file.
    """
    # Create a list to hold the ONNX graph nodes
    nodes = []
    initializers = []
    inputs = []
    outputs = []

    # Get indices for input and output
    input_indices = nn["NeuralNetworkModel"]["inputEntryIndices"]
    output_indices = nn["NeuralNetworkModel"]["outputEntryIndices"]

    # Read variable names from CSV file
    csv_filename = nn["NeuralNetworkModel"]["InputFileName"]
    csv_path = f"data/input/{csv_filename}"
    delimiter = nn["NeuralNetworkModel"].get("Delimiter", ",")

    variable_names = []
    try:
        import csv
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            headers = next(reader)  # Read first row (headers)
            variable_names = headers
    except Exception as e:
        print(f"Warning: Could not read variable names from {csv_path}: {e}")
        # Fallback to generic names
        variable_names = [f"input_{i}" for i in range(len(input_indices) + len(output_indices))]
    
    # Get min/max values for normalization (same values used during training)
    min_data = database['min_data']
    max_data = database['max_data']
    
    # Extract min/max for input and output using the same global normalization
    input_min = min_data[input_indices]
    input_max = max_data[input_indices]
    output_min = min_data[output_indices]
    output_max = max_data[output_indices]
    
    # Calculate normalization parameters
    input_range = input_max - input_min
    output_range = output_max - output_min
    
    # Define input and output names with consistent naming scheme
    input_var_names = [variable_names[i] for i in input_indices]
    output_var_names = [variable_names[i] for i in output_indices]

    # Always use descriptive names regardless of count
    input_name = "_".join(input_var_names) if len(input_var_names) <= 8 else f"inputs_{len(input_var_names)}_vars"
    final_output_name = "_".join(output_var_names) if len(output_var_names) <= 10 else f"outputs_{len(output_var_names)}_vars"

    # Define the input tensor
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    inputs.append(input_tensor)

    # Add normalization constants as initializers (only the ones we actually use)
    initializers.append(helper.make_tensor('input_min', TensorProto.FLOAT, input_min.shape, input_min.flatten()))
    initializers.append(helper.make_tensor('input_range', TensorProto.FLOAT, input_range.shape, input_range.flatten()))
    
    initializers.append(helper.make_tensor('output_min', TensorProto.FLOAT, output_min.shape, output_min.flatten()))
    initializers.append(helper.make_tensor('output_range', TensorProto.FLOAT, output_range.shape, output_range.flatten()))

    # INPUT NORMALIZATION: (data - min_data) / (max_data - min_data)
    # Step 1: Subtract min from input
    sub_node = helper.make_node(
        'Sub',
        [input_name, 'input_min'],
        ['input_sub_min'],
        name='input_subtract_min'
    )
    nodes.append(sub_node)
    
    # Step 2: Divide by range to get normalized input
    div_node = helper.make_node(
        'Div',
        ['input_sub_min', 'input_range'],
        ['normalized_input'],
        name='input_normalize'
    )
    nodes.append(div_node)

    # NEURAL NETWORK LAYERS
    # Iterate through the layers to create ONNX nodes
    prev_output = 'normalized_input'

    for i, (layer_name, layer) in enumerate(network_layers.items()):
        weight_name = f"{layer_name}_W"
        bias_name = f"{layer_name}_B"
        output_name = f"{layer_name}_out"

        # Add weights and biases as initializers
        initializers.append(helper.make_tensor(weight_name, TensorProto.FLOAT, params_values[f"W{i+1}"].shape, params_values[f"W{i+1}"].flatten()))
        initializers.append(helper.make_tensor(bias_name, TensorProto.FLOAT, params_values[f"b{i+1}"].shape, params_values[f"b{i+1}"].flatten()))

        # Create a MatMul node
        matmul_node = helper.make_node(
            'MatMul',
            [prev_output, weight_name],
            [f"{layer_name}_matmul_out"],
            name=f"{layer_name}_matmul"
        )
        nodes.append(matmul_node)

        # Create an Add node for the bias
        add_node = helper.make_node(
            'Add',
            [f"{layer_name}_matmul_out", bias_name],
            [f"{layer_name}_add_out"],
            name=f"{layer_name}_add"
        )
        nodes.append(add_node)

        # Add activation function node if specified
        if layer['activation'].lower() == 'relu':
            activation_node = helper.make_node(
                'Relu',
                [f"{layer_name}_add_out"],
                [output_name],
                name=f"{layer_name}_relu"
            )
        elif layer['activation'].lower() == 'elu':
            activation_node = helper.make_node(
                'Elu',
                [f"{layer_name}_add_out"],
                [output_name],
                name=f"{layer_name}_elu"
            )
        elif layer['activation'].lower() == 'sigmoid':
            activation_node = helper.make_node(
                'Sigmoid',
                [f"{layer_name}_add_out"],
                [output_name],
                name=f"{layer_name}_sigmoid"
            )
        elif layer['activation'].lower() == 'tanh':
            activation_node = helper.make_node(
                'Tanh',
                [f"{layer_name}_add_out"],
                [output_name],
                name=f"{layer_name}_tanh"
            )
        else:
            raise ValueError(f"Unsupported activation function: {layer['activation']}")

        nodes.append(activation_node)
        prev_output = output_name

    # OUTPUT DENORMALIZATION: normalized_output * (max_data - min_data) + min_data
    # Step 1: Multiply normalized output by range
    mul_node = helper.make_node(
        'Mul',
        [prev_output, 'output_range'],
        ['output_mul_range'],
        name='output_multiply_range'
    )
    nodes.append(mul_node)
    
    # Step 2: Add min to get denormalized output
    add_min_node = helper.make_node(
        'Add',
        ['output_mul_range', 'output_min'],
        [final_output_name],
        name='output_add_min'
    )
    nodes.append(add_min_node)

    # Define the output tensor
    output_tensor = helper.make_tensor_value_info(final_output_name, TensorProto.FLOAT, [None, len(output_indices)])
    outputs.append(output_tensor)

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'NeuralNetworkGraph',
        inputs,
        outputs,
        initializers
    )

    # Create the model
    model = helper.make_model(graph, producer_name='custom_nn')
    onnx.save(model, output_file)