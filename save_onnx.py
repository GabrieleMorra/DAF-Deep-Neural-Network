import onnx
import numpy as np
from onnx import helper, TensorProto

def save_onnx(params_values, network_layers, input_shape, output_file):
    """
    Save the neural network in ONNX file format.

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

    # Define the input tensor
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    inputs.append(input_tensor)

    # Iterate through the layers to create ONNX nodes
    prev_output = 'input'
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

    # Define the output tensor
    output_tensor = helper.make_tensor_value_info(prev_output, TensorProto.FLOAT, [None, network_layers[list(network_layers.keys())[-1]]['output_dim']])
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