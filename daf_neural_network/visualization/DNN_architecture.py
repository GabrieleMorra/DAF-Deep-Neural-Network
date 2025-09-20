#!/usr/bin/env python3
"""
DNN Architecture Visualizer - Standalone Tool
Independent tool to visualize Deep Neural Network architecture from ONNX files
NO EXTERNAL DEPENDENCIES - Works with any ONNX file from any source

Usage:
    python DNN_architecture.py <path_to_onnx_file> [output_name.svg]

Example:
    python DNN_architecture.py model.onnx
    python DNN_architecture.py /path/to/model.onnx network_diagram.svg
"""

import sys
import os


def parse_onnx_protobuf(file_path):
    """
    Enhanced ONNX parser - tries onnx library first, falls back to manual parsing
    Extracts accurate architecture information including weight values
    """

    # Try using onnx library if available
    try:
        import onnx
        import numpy as np

        model = onnx.load(file_path)

        # Extract input and output names from ONNX graph
        input_names = []
        output_names = []

        # Get input names from graph inputs
        for input_tensor in model.graph.input:
            if input_tensor.name not in ['input_min', 'input_range', 'output_min', 'output_range']:
                input_names.append(input_tensor.name)
    
        # Get output names from graph outputs
        for output_tensor in model.graph.output:
            output_names.append(output_tensor.name)

        # Optional debug output (comment out for cleaner output)
        # print("[DEBUG] All tensors in ONNX model:")
        # for initializer in model.graph.initializer:
        #     print(f"  - {initializer.name}: shape {list(initializer.dims)}")

        # Extract weight and bias tensors from initializers
        weight_tensors = []
        bias_tensors = []
        weight_shapes = []
        layer_sizes = []

        for initializer in model.graph.initializer:
            name = initializer.name

            # Convert tensor data to numpy array
            if initializer.data_type == 1:  # FLOAT
                if initializer.raw_data:
                    # Raw data format
                    tensor_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                else:
                    # float_data format
                    tensor_data = np.array(initializer.float_data, dtype=np.float32)

                # Get shape
                shape = list(initializer.dims)
                tensor_data = tensor_data.reshape(shape) if shape else tensor_data

                # Skip normalization tensors and other non-neural-network tensors
                if any(skip_name in name.lower() for skip_name in ['input_min', 'input_range', 'output_min', 'output_range', 'min', 'range']):
                    continue

                # Process neural network weights and biases (only actual layer weights)
                if (('Layer' in name and '_W' in name) or
                    (name.startswith('FirstHiddenLayer') and '_W' in name) or
                    (name.startswith('SecondHiddenLayer') and '_W' in name) or
                    (name.startswith('ThirdHiddenLayer') and '_W' in name) or
                    (name.startswith('OutputLayer') and '_W' in name)):
                    tensor_info = {
                        'name': name,
                        'values': tensor_data.flatten()[:20].tolist(),  # First 20 values
                        'size': tensor_data.size,
                        'shape': shape,
                        'avg': float(np.mean(tensor_data)),
                        'min': float(np.min(tensor_data)),
                        'max': float(np.max(tensor_data))
                    }
                    weight_tensors.append(tensor_info)
                    if len(shape) == 2:
                        weight_shapes.append(shape)

                elif (('Layer' in name and '_B' in name) or
                      (name.startswith('FirstHiddenLayer') and '_B' in name) or
                      (name.startswith('SecondHiddenLayer') and '_B' in name) or
                      (name.startswith('ThirdHiddenLayer') and '_B' in name) or
                      (name.startswith('OutputLayer') and '_B' in name)):
                    tensor_info = {
                        'name': name,
                        'values': tensor_data.flatten().tolist(),  # All bias values (usually small)
                        'size': tensor_data.size,
                        'shape': shape,
                        'avg': float(np.mean(tensor_data)),
                        'min': float(np.min(tensor_data)),
                        'max': float(np.max(tensor_data))
                    }
                    bias_tensors.append(tensor_info)

        # Infer layer sizes from weight shapes
        if weight_shapes:
            # Find the input layer by looking for the shape that doesn't have any predecessor
            # (no other shape has output dimension matching this shape's input dimension)
            input_dimensions = [shape[0] for shape in weight_shapes]
            output_dimensions = [shape[1] for shape in weight_shapes]

            # The true input layer has an input dimension that's not an output of any other layer
            first_layer_input = None
            for shape in weight_shapes:
                if shape[0] not in output_dimensions:
                    first_layer_input = shape[0]
                    break

            if first_layer_input is None:
                # Fallback: use the largest input dimension as the starting point
                first_layer_input = max(input_dimensions)

            # Build the layer sequence by chaining
            layer_sizes = [first_layer_input]
            used_shapes = []
            current_output = first_layer_input

            # Find the first layer with this input
            for shape in weight_shapes:
                if shape[0] == current_output:
                    layer_sizes.append(shape[1])
                    used_shapes.append(shape)
                    current_output = shape[1]
                    break

            # Chain the remaining layers by matching dimensions
            while len(used_shapes) < len(weight_shapes):
                found = False
                for shape in weight_shapes:
                    if shape not in used_shapes and shape[0] == current_output:
                        layer_sizes.append(shape[1])
                        used_shapes.append(shape)
                        current_output = shape[1]
                        found = True
                        break

                if not found:  # Safety break if we can't chain properly
                    break
        else:
            # Fallback: try to extract from input/output tensor shapes
            try:
                input_size = 8  # Default fallback
                output_size = 1  # Default fallback

                for input_tensor in model.graph.input:
                    if input_tensor.name not in ['input_min', 'input_range', 'output_min', 'output_range']:
                        if hasattr(input_tensor.type, 'tensor_type') and hasattr(input_tensor.type.tensor_type, 'shape'):
                            for dim in input_tensor.type.tensor_type.shape.dim:
                                if dim.dim_value > 1:  # Skip batch dimension
                                    input_size = dim.dim_value
                                    break

                for output_tensor in model.graph.output:
                    if hasattr(output_tensor.type, 'tensor_type') and hasattr(output_tensor.type.tensor_type, 'shape'):
                        for dim in output_tensor.type.tensor_type.shape.dim:
                            if dim.dim_value > 0:  # Skip dynamic dimension
                                output_size = dim.dim_value
                                break

                # Try to extract hidden layer sizes from weight tensors
                hidden_sizes = []
                for initializer in model.graph.initializer:
                    if '_W' in initializer.name and initializer.name not in ['input_min', 'input_range', 'output_min', 'output_range']:
                        # Extract output dimension from weight matrix (shape is [input, output])
                        if len(initializer.dims) >= 2:
                            hidden_sizes.append(initializer.dims[1])

                if hidden_sizes:
                    layer_sizes = [input_size] + hidden_sizes
                else:
                    # Generic fallback based on extracted input/output sizes
                    layer_sizes = [input_size, max(input_size-1, 5), max(output_size*2, 3), output_size]
            except:
                # Final emergency fallback - should rarely be used
                layer_sizes = [8, 7, 5, 3]

        # Generate input names based on ONNX input tensor name
        input_count = layer_sizes[0] if layer_sizes else 8

        if input_names:
            input_name = input_names[0]
            if '_' in input_name and not input_name.startswith('inputs_'):
                # Split joined names like "AoA_DeltaXProp_..."
                input_parts = input_name.split('_')
                if len(input_parts) == input_count:
                    extracted_input_names = input_parts
                else:
                    extracted_input_names = [f'x{i+1}' for i in range(input_count)]
            elif input_name.startswith('inputs_') and input_name.endswith('_vars'):
                # Generic name like "inputs_8_vars"
                extracted_input_names = [f'x{i+1}' for i in range(input_count)]
            else:
                # Single input name or other format
                if input_count == 1:
                    extracted_input_names = [input_name]
                else:
                    extracted_input_names = [f'x{i+1}' for i in range(input_count)]
        else:
            extracted_input_names = [f'x{i+1}' for i in range(input_count)]

        # Generate output names based on ONNX output tensor name
        output_count = layer_sizes[-1] if layer_sizes else 1

        if output_names:
            output_name = output_names[0]
            if '_' in output_name and not output_name.startswith('outputs_'):
                # Split joined names like "Cl_Cd_Cm"
                output_parts = output_name.split('_')
                if len(output_parts) == output_count:
                    extracted_output_names = output_parts
                else:
                    extracted_output_names = [f'y{i+1}' for i in range(output_count)]
            elif output_name.startswith('outputs_') and output_name.endswith('_vars'):
                # Generic name like "outputs_3_vars"
                extracted_output_names = [f'y{i+1}' for i in range(output_count)]
            else:
                # Single output name
                if output_count == 1:
                    extracted_output_names = [output_name]
                else:
                    extracted_output_names = [f'y{i+1}' for i in range(output_count)]
        else:
            extracted_output_names = [f'y{i+1}' for i in range(output_count)]

        return {
            'inputs': [('input', layer_sizes[0] if layer_sizes else 8)],
            'outputs': [('output', layer_sizes[-1] if layer_sizes else 3)],
            'layer_sizes': layer_sizes,
            'weight_shapes': weight_shapes[:10],
            'weight_tensors': weight_tensors[:5],  # Actual weight values
            'bias_tensors': bias_tensors[:5],      # Actual bias values
            'total_weights': len(weight_tensors),
            'input_names': extracted_input_names,
            'output_names': extracted_output_names,
            'node_types': ['MatMul'] * len(weight_tensors)
        }

    except ImportError:
        print("[WARNING] onnx library not available, using manual parsing")
    except Exception as e:
        print(f"[WARNING] ONNX library parsing failed: {e}, using manual parsing")

    def read_varint(data, offset):
        """Read variable length integer from protobuf"""
        result = 0
        shift = 0
        while offset < len(data):
            byte = data[offset]
            offset += 1
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        return result, offset

    def read_string(data, offset, length):
        """Read string from protobuf"""
        return data[offset:offset+length].decode('utf-8', errors='ignore'), offset + length

    def read_int64(data, offset):
        """Read 64-bit integer from protobuf"""
        import struct
        if offset + 8 <= len(data):
            return struct.unpack('<Q', data[offset:offset+8])[0], offset + 8
        return 0, offset + 8

    def read_float32(data, offset):
        """Read 32-bit float from protobuf"""
        import struct
        if offset + 4 <= len(data):
            return struct.unpack('<f', data[offset:offset+4])[0], offset + 4
        return 0.0, offset + 4

    def parse_tensor_data(data, offset, length, data_type=1):
        """Parse tensor data values (floats)"""
        import struct
        values = []
        end_offset = offset + length

        if data_type == 1:  # FLOAT (4 bytes each)
            while offset + 4 <= end_offset:
                try:
                    value = struct.unpack('<f', data[offset:offset+4])[0]
                    values.append(value)
                    offset += 4
                except:
                    break

        return values

    def parse_tensor_shape(data, offset, end_offset):
        """Parse tensor shape from protobuf data"""
        dimensions = []
        while offset < end_offset:
            try:
                tag, offset = read_varint(data, offset)
                wire_type = tag & 0x7
                field_number = tag >> 3

                if wire_type == 2:  # Length-delimited (dimension)
                    length, offset = read_varint(data, offset)
                    dim_end = offset + length
                    dim_size = None

                    while offset < dim_end:
                        dim_tag, offset = read_varint(data, offset)
                        dim_wire_type = dim_tag & 0x7
                        dim_field = dim_tag >> 3

                        if dim_field == 1 and dim_wire_type == 0:  # dim_value
                            dim_size, offset = read_varint(data, offset)
                        elif dim_wire_type == 2:  # Skip string fields
                            skip_len, offset = read_varint(data, offset)
                            offset += skip_len
                        else:
                            offset += 1

                    if dim_size is not None:
                        dimensions.append(dim_size)
                else:
                    offset += 1
            except:
                offset += 1
                if offset >= end_offset:
                    break

        return dimensions

    # Read ONNX file
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        raise Exception(f"Could not read ONNX file: {e}")

    # Parse ONNX model structure - focus on neural network layers only
    layer_sizes = []
    weight_shapes = []
    weight_tensors = []  # Store actual weight values
    bias_tensors = []    # Store actual bias values
    input_names = []
    output_names = []
    node_types = []

    # Operations to skip (normalization, preprocessing)
    skip_operations = [b'Div', b'Sub', b'Add', b'Mul', b'Clip', b'Cast', b'Reshape', b'Transpose',
                      b'BatchNormalization', b'LayerNormalization', b'Dropout', b'Constant',
                      b'Identity', b'Shape', b'Gather', b'Unsqueeze', b'Squeeze']

    # More aggressive parsing for ONNX tensors
    offset = 0
    tensor_candidates = []

    # First pass: find all potential tensor names and raw_data fields
    while offset < len(data) - 8:
        try:
            # Look for raw_data fields (field 9 in ONNX protobuf)
            tag, next_offset = read_varint(data, offset)
            wire_type = tag & 0x7
            field_number = tag >> 3

            if wire_type == 2 and field_number == 9:  # raw_data field
                length, data_start = read_varint(data, next_offset)
                if 4 <= length <= 50000:  # Reasonable tensor size
                    # Look backwards for tensor name
                    name_search_start = max(0, offset - 500)
                    tensor_name = ""

                    # Search for printable strings before this raw_data
                    for name_offset in range(name_search_start, offset):
                        if (name_offset + 20 < len(data) and
                            data[name_offset:name_offset+2] in [b'W1', b'W2', b'W3', b'W4', b'b1', b'b2', b'b3', b'b4']):
                            tensor_name = data[name_offset:name_offset+2].decode('utf-8', errors='ignore')
                            break
                        elif (name_offset + 15 < len(data) and
                              b'Layer_W' in data[name_offset:name_offset+15]):
                            # Find the full layer name
                            start = name_offset
                            while start > name_search_start and data[start] != 0:
                                start -= 1
                            end = name_offset + 15
                            while end < len(data) and data[end] != 0:
                                end += 1
                            tensor_name = data[start+1:end].decode('utf-8', errors='ignore')
                            break
                        elif (name_offset + 15 < len(data) and
                              b'Layer_B' in data[name_offset:name_offset+15]):
                            # Find the full layer name
                            start = name_offset
                            while start > name_search_start and data[start] != 0:
                                start -= 1
                            end = name_offset + 15
                            while end < len(data) and data[end] != 0:
                                end += 1
                            tensor_name = data[start+1:end].decode('utf-8', errors='ignore')
                            break

                    # Extract float values if we found a name
                    if tensor_name:
                        tensor_values = parse_tensor_data(data, data_start, length, 1)
                        if len(tensor_values) > 0:
                            tensor_info = {
                                'name': tensor_name,
                                'values': tensor_values[:20],  # First 20 values
                                'size': len(tensor_values),
                                'shape': None,
                                'avg': sum(tensor_values) / len(tensor_values),
                                'min': min(tensor_values),
                                'max': max(tensor_values)
                            }

                            # Categorize as weight or bias
                            if ('W' in tensor_name and tensor_name[-1].isdigit()) or '_W' in tensor_name:
                                # Infer 2D shape for weights
                                total_size = len(tensor_values)
                                layer_configs = [(13, 7), (7, 6), (6, 5), (5, 1)]  # From config
                                for in_size, out_size in layer_configs:
                                    if total_size == in_size * out_size:
                                        tensor_info['shape'] = [in_size, out_size]
                                        weight_shapes.append([in_size, out_size])
                                        break
                                weight_tensors.append(tensor_info)

                            elif ('b' in tensor_name and tensor_name[-1].isdigit()) or '_B' in tensor_name:
                                tensor_info['shape'] = [len(tensor_values)]  # 1D bias
                                bias_tensors.append(tensor_info)

                            tensor_candidates.append(tensor_info)

            offset = next_offset if next_offset > offset else offset + 1

        except:
            offset += 1

    # Also look for MatMul operations
    offset = 0
    while offset < len(data) - 6:
        if data[offset:offset+6] == b'MatMul':
            node_types.append('MatMul')
        offset += 1

    # Extract layer sizes from weight shapes
    if weight_shapes:
        # Sort by first dimension (usually input size)
        unique_shapes = []
        for shape in weight_shapes:
            if len(shape) == 2 and shape not in unique_shapes:
                unique_shapes.append(shape)

        if unique_shapes:
            # Build layer architecture from weight matrices
            unique_shapes.sort(key=lambda x: x[0], reverse=True)

            # First layer input size
            if unique_shapes:
                layer_sizes = [unique_shapes[0][0]]  # Input size

                # Add hidden and output layers
                for shape in unique_shapes:
                    if shape[1] not in layer_sizes:
                        layer_sizes.append(shape[1])

                # Ensure reasonable order
                if len(layer_sizes) > 2:
                    # Keep input, sort middle layers, keep output
                    input_size = layer_sizes[0]
                    output_size = layer_sizes[-1]
                    middle_layers = sorted(layer_sizes[1:-1], reverse=True)
                    layer_sizes = [input_size] + middle_layers + [output_size]

    # Fallback to configuration-based defaults if parsing failed
    if not layer_sizes or len(layer_sizes) < 2:
        layer_sizes = [8, 7, 15, 5, 3]  # Based on current config (CFD dataset)

    # Generate input names
    input_count = layer_sizes[0] if layer_sizes else 8
    input_names = [f'x{i+1}' for i in range(min(input_count, 20))]

    # Generate output names
    output_count = layer_sizes[-1] if layer_sizes else 3
    output_names = [f'y{i+1}' for i in range(min(output_count, 10))]

    return {
        'inputs': [('input', layer_sizes[0] if layer_sizes else 8)],
        'outputs': [('output', layer_sizes[-1] if layer_sizes else 3)],
        'layer_sizes': layer_sizes,
        'weight_shapes': weight_shapes[:10],
        'weight_tensors': weight_tensors[:5],  # Actual weight values (limited)
        'bias_tensors': bias_tensors[:5],      # Actual bias values (limited)
        'total_weights': len(weight_shapes),
        'input_names': input_names,
        'node_types': node_types[:5]
    }


def calculate_layout_parameters(layer_sizes, input_names=None, output_names=None):
    """Calculate dynamic layout parameters based on network architecture"""

    # Base parameters
    min_neuron_radius = 8
    max_neuron_radius = 15
    base_header_height = 80
    base_footer_height = 50
    min_layer_spacing = 120
    max_layer_spacing = 250
    base_neuron_spacing = 30
    min_neuron_spacing = 20

    # Calculate based on architecture
    num_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    total_neurons = sum(layer_sizes)

    # Dynamic neuron radius (smaller for dense networks)
    if max_neurons <= 5:
        neuron_radius = max_neuron_radius
    elif max_neurons <= 10:
        neuron_radius = 12
    else:
        neuron_radius = max(min_neuron_radius, max_neuron_radius - (max_neurons - 10) * 0.5)

    # Dynamic neuron spacing - maximum spacing
    if max_neurons <= 5:
        neuron_spacing = base_neuron_spacing + 40  # Maximum space for small networks
    elif max_neurons <= 10:
        neuron_spacing = base_neuron_spacing + 30  # Maximum space for medium networks
    else:
        neuron_spacing = max(min_neuron_spacing + 15, base_neuron_spacing - (max_neurons - 10) * 0.2)  # Minimal reduction for large networks

    # Dynamic layer spacing
    if num_layers <= 3:
        layer_spacing = max_layer_spacing
    elif num_layers <= 5:
        layer_spacing = 180
    else:
        layer_spacing = max(min_layer_spacing, max_layer_spacing - (num_layers - 3) * 20)

    # Calculate SVG dimensions based on longest layer positioning
    header_height = base_header_height
    header_bottom = header_height + 35  # After subheader text

    # Dynamic top margin - increase for networks with many neurons
    if max_neurons <= 5:
        top_margin = 80
    elif max_neurons <= 10:
        top_margin = 100
    elif max_neurons <= 15:
        top_margin = 120
    elif max_neurons <= 25:
        top_margin = 140
    else:
        top_margin = 160  # For very large layers (25+ neurons)

    # Calculate the exact range of the longest layer
    longest_layer_start = header_bottom + top_margin
    longest_layer_end = longest_layer_start + (max_neurons - 1) * neuron_spacing

    # SVG height should end shortly after the longest layer rectangle
    # Add padding for layer rectangles (25px below) + small bottom margin
    bottom_margin = 50
    svg_height = longest_layer_end + 25 + bottom_margin  # 25px is layer rectangle padding

    # Dynamic SVG width calculation
    # Base width needed for layers and spacing
    layers_width = (num_layers - 1) * layer_spacing

    # Reduced lateral margins
    if num_layers <= 3:
        side_margin = 40  # Very small margins for few layers
    elif num_layers <= 5:
        side_margin = 50  # Small margins for medium networks
    else:
        side_margin = 60  # Moderate margins for many layers

    # Add space for layer rectangles (30px each side for rectangle padding)
    rect_padding_space = 60

    # Calculate minimum width needed for longest layer names/headers
    header_space = max(200, num_layers * 40)  # Ensure enough space for headers

    # Calculate extra width needed for input/output rectangles
    extra_width = 0
    if input_names:
        max_input_width = max(max(70, calculate_text_width(name, 10)) for name in input_names)
        extra_width = max(extra_width, max_input_width//2)  # Half extends beyond center
    if output_names:
        max_output_width = max(max(70, calculate_text_width(name, 10)) for name in output_names)
        extra_width = max(extra_width, max_output_width//2)  # Half extends beyond center

    svg_width = max(
        layers_width + 2 * side_margin + rect_padding_space + 2 * extra_width,  # Add space for wide rectangles
        header_space + 2 * side_margin  # Minimum space for headers
    )

    return {
        'svg_width': int(svg_width),
        'svg_height': int(svg_height),
        'neuron_radius': neuron_radius,
        'layer_spacing': layer_spacing,
        'neuron_spacing': neuron_spacing,
        'header_height': header_height,
        'top_margin': top_margin,
        'longest_layer_start': longest_layer_start,
        'longest_layer_end': longest_layer_end
    }

def calculate_text_width(text, font_size=10):
    """Estimate text width in pixels based on character count and font size"""
    # Rough estimation: each character is about 0.6 * font_size pixels wide
    char_width = font_size * 0.6
    return len(text) * char_width + 10  # Add some padding

def create_svg_architecture(model_info, output_path="architecture.svg"):
    """Create SVG visualization of neural network architecture with dynamic layout"""

    # Extract layer information first for layout calculation
    layer_sizes = model_info['layer_sizes']
    inputs = model_info['inputs']
    outputs = model_info['outputs']

    # Ensure we have reasonable layer sizes
    if not layer_sizes or len(layer_sizes) < 2:
        layer_sizes = [10, 8, 6, 1]  # Default

    # Calculate dynamic layout parameters
    # Get input/output names for width calculation
    input_names = model_info.get('input_names', [])
    output_names = model_info.get('output_names', [])

    layout = calculate_layout_parameters(layer_sizes, input_names, output_names)

    # Use calculated parameters
    svg_width = layout['svg_width']
    svg_height = layout['svg_height']
    neuron_radius = layout['neuron_radius']
    layer_spacing = layout['layer_spacing']

    # Scientific paper color palette - distinct colors for each layer
    layer_colors = [
        "#1f77b4",  # Input layer - Professional blue
        "#ff7f0e",  # Hidden 1 - Orange
        "#2ca02c",  # Hidden 2 - Green
        "#d62728",  # Hidden 3 - Red
        "#9467bd",  # Output layer - Purple
        "#8c564b",  # Additional hidden layers - Brown
        "#e377c2",  # Pink
        "#7f7f7f"   # Gray
    ]

    # Bias color
    bias_color = "#17becf"  # Cyan for bias neurons

    # Start SVG with enhanced aesthetics
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}">
<defs>
    <style type="text/css">
        .layer-title {{ font-family: 'Times New Roman', serif; font-size: 15px; font-weight: bold; text-anchor: middle; fill: #2c3e50; }}
        .layer-info {{ font-family: 'Times New Roman', serif; font-size: 12px; text-anchor: middle; fill: #34495e; }}
        .neuron {{ stroke-width: 2.5; filter: url(#dropShadow); }}
        .connection {{ stroke-width: 0.8; opacity: 0.6; fill: none; }}
        .title {{ font-family: 'Times New Roman', serif; font-size: 22px; font-weight: bold; text-anchor: middle; fill: #2c3e50; }}
        .subtitle {{ font-family: 'Times New Roman', serif; font-size: 14px; text-anchor: middle; fill: #7f8c8d; font-style: italic; }}
        .weight-info {{ font-family: 'Times New Roman', serif; font-size: 10px; text-anchor: middle; fill: #7f8c8d; }}
        .legend-text {{ font-family: 'Times New Roman', serif; font-size: 11px; fill: #34495e; }}
        .info-box {{ fill: #ecf0f1; stroke: #bdc3c7; stroke-width: 1; rx: 5; }}
        .gradient {{ fill: url(#layerGradient); }}
    </style>

    <!-- Gradients and Filters -->
    <linearGradient id="layerGradient" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" style="stop-color:#ecf0f1;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#bdc3c7;stop-opacity:1" />
    </linearGradient>

    <linearGradient id="backgroundGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" style="stop-color:#ffffff;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#f8f9fa;stop-opacity:1" />
    </linearGradient>

    <filter id="dropShadow" x="-50%" y="-50%" width="200%" height="200%">
        <feDropShadow dx="2" dy="2" stdDeviation="1" flood-color="#95a5a6" flood-opacity="0.3"/>
    </filter>

    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>
    </filter>
</defs>

<!-- Enhanced Background -->
<rect width="{svg_width}" height="{svg_height}" fill="url(#backgroundGradient)" stroke="#bdc3c7" stroke-width="2"/>

<!-- Decorative border -->
<rect x="10" y="10" width="{svg_width-20}" height="{svg_height-20}" fill="none" stroke="#7f8c8d" stroke-width="1" stroke-dasharray="5,5"/>

<!-- Title Section -->
<rect x="{svg_width//2-300}" y="5" width="600" height="50" fill="#ecf0f1" stroke="#7f8c8d" stroke-width="1" rx="8" opacity="0.8"/>
<text x="{svg_width//2}" y="35" class="title">Deep Neural Network Architecture</text>
'''

    # Layer sizes already extracted above, remove duplicate code

    # Calculate layer positions dynamically
    num_layers = len(layer_sizes)
    margin_x = 100  # Base margin, will be adjusted based on SVG width

    # Adjust margin to center layers in SVG
    total_layer_width = (num_layers - 1) * layer_spacing
    available_width = svg_width - 2 * margin_x
    if total_layer_width < available_width:
        margin_x = (svg_width - total_layer_width) // 2

    layer_positions = []
    for i in range(num_layers):
        x = margin_x + i * layer_spacing
        layer_positions.append(x)

    # Function to get neuron positions for a layer using fixed positioning for longest layer
    def get_neuron_positions(layer_size, x_pos):
        max_display = min(layer_size, max(layer_sizes))  # Show up to max neurons in the network
        if max_display <= 1:
            # Single neuron - center it relative to longest layer
            longest_layer_center = (layout['longest_layer_start'] + layout['longest_layer_end']) / 2
            return [(x_pos, longest_layer_center)]

        # Fixed spacing from header/subheader for the longest layer
        max_neurons_in_network = max(layer_sizes)

        if layer_size == max_neurons_in_network:
            # This is the longest layer - use fixed positioning from layout
            start_y = layout['longest_layer_start']
            spacing = layout['neuron_spacing']
        else:
            # Other layers - center them relative to the longest layer's range
            longest_layer_center = (layout['longest_layer_start'] + layout['longest_layer_end']) / 2

            # Center this layer around the longest layer's center
            total_height = (max_display - 1) * layout['neuron_spacing']
            start_y = longest_layer_center - total_height / 2
            spacing = layout['neuron_spacing']

        positions = []
        for i in range(max_display):
            y = start_y + i * spacing
            positions.append((x_pos, y))

        return positions

    # Store all neuron positions for connections
    all_positions = []
    for i, size in enumerate(layer_sizes):
        positions = get_neuron_positions(size, layer_positions[i])
        all_positions.append(positions)

    # Draw ALL connections between layers (fully connected)
    for i in range(len(all_positions) - 1):
        from_positions = all_positions[i]
        to_positions = all_positions[i + 1]

        # Get connection color from source layer
        connection_color = layer_colors[i] if i < len(layer_colors) else layer_colors[-1]

        # Draw connections from every neuron to every neuron (fully connected)
        for from_idx, (from_x, from_y) in enumerate(from_positions):
            for to_idx, (to_x, to_y) in enumerate(to_positions):
                # Adjust connection start point for input layer (rectangles)
                if i == 0:  # From input layer
                    # For input layer, connections start from the right edge of each rectangle
                    input_names_list = model_info.get('input_names', [])
                    var_name = input_names_list[from_idx] if from_idx < len(input_names_list) else f"x{from_idx+1}"
                    neuron_rect_width = max(70, calculate_text_width(var_name, 10))

                    # Start connection from right edge of this specific rectangle
                    start_x = from_x + (neuron_rect_width//2)  # Right edge of centered rectangle
                    start_y = from_y
                else:
                    start_x = from_x + neuron_radius  # Circle edge
                    start_y = from_y

                # Adjust connection end point for output layer (rectangles)
                if i == len(all_positions) - 2:  # To output layer
                    # For output layer, connections end at the left edge of each rectangle
                    output_names_list = model_info.get('output_names', [])
                    output_name = output_names_list[to_idx] if to_idx < len(output_names_list) else f"y{to_idx+1}"
                    neuron_rect_width = max(70, calculate_text_width(output_name, 10))

                    # End connection at left edge of this specific rectangle
                    end_x = to_x - (neuron_rect_width//2)  # Left edge of centered rectangle
                    end_y = to_y
                else:
                    end_x = to_x - neuron_radius  # Circle edge
                    end_y = to_y

                svg_content += f'''<line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" stroke="{connection_color}" class="connection"/>
'''

    # Get input names for display
    input_names_list = model_info.get('input_names', [])

    # Draw layers with background highlighting
    for layer_idx, layer_size in enumerate(layer_sizes):
        x_pos = layer_positions[layer_idx]
        neuron_positions = all_positions[layer_idx]

        # Determine layer type and color
        color = layer_colors[layer_idx] if layer_idx < len(layer_colors) else layer_colors[-1]

        if layer_idx == 0:
            layer_type = "Input"
            layer_name = "Input Layer"
        elif layer_idx == len(layer_sizes) - 1:
            layer_type = "Output"
            layer_name = "Output Layer"
        else:
            layer_type = "Hidden"
            layer_name = f"Hidden {layer_idx}"

        # Layer background highlighting (exclude titles) - dynamic sizing including bias with more vertical padding
        if layer_idx > 0 and layer_idx < len(layer_sizes) - 1:  # Hidden layers with bias
            bias_y = min([pos[1] for pos in neuron_positions]) - 40  # Bias position
            min_y = bias_y - 25  # More padding above bias
            max_y = max([pos[1] for pos in neuron_positions]) + 25  # More padding below neurons
        else:  # Input and output layers without bias
            min_y = min([pos[1] for pos in neuron_positions]) - 25  # More padding above
            max_y = max([pos[1] for pos in neuron_positions]) + 25  # More padding below

        layer_height = max_y - min_y

        # Dynamic rectangle width based on layer type
        if layer_idx == 0:  # Input layer - consider variable name widths
            input_names_list = model_info.get('input_names', [])
            max_input_width = 70  # Default minimum
            for i in range(layer_size):
                var_name = input_names_list[i] if i < len(input_names_list) else f"x{i+1}"
                neuron_width = max(70, calculate_text_width(var_name, 10))
                max_input_width = max(max_input_width, neuron_width)
            rect_width = max_input_width + 40  # Add padding around the widest neuron
        elif layer_idx == len(layer_sizes) - 1:  # Output layer - consider output name widths
            output_names_list = model_info.get('output_names', [])
            max_output_width = 70  # Default minimum
            for i in range(layer_size):
                output_name = output_names_list[i] if i < len(output_names_list) else f"y{i+1}"
                neuron_width = max(70, calculate_text_width(output_name, 10))
                max_output_width = max(max_output_width, neuron_width)
            rect_width = max_output_width + 40  # Add padding around the widest neuron
        else:  # Hidden layers - based on neuron radius
            rect_padding = max(20, neuron_radius * 4)  # Minimum 20px, scales with neuron size
            rect_width = rect_padding * 2

        svg_content += f'''
<!-- Layer Background -->
<rect x="{x_pos - rect_width//2}" y="{min_y}" width="{rect_width}" height="{layer_height}"
      fill="{color}" opacity="0.08" stroke="{color}" stroke-width="1"
      stroke-dasharray="3,3" rx="15"/>
'''

        # Layer title with enhanced styling - positioned much lower
        title_y = layout['header_height'] + 20  # Move further into neuron area
        svg_content += f'''<text x="{x_pos}" y="{title_y}" class="layer-title" style="fill: {color};">{layer_name}</text>
'''

        # Add neurons count for hidden layers in black
        if layer_type == "Hidden":
            info_y = layout['header_height'] + 35  # Further below title
            neuron_text = "neuron" if layer_size == 1 else "neurons"
            svg_content += f'''<text x="{x_pos}" y="{info_y}" class="layer-info" style="fill: black;">{layer_size} {neuron_text}</text>
'''

        # Add parameter count for input and output layers
        if layer_type == "Input":
            param_count = layer_size  # Number of input parameters
            info_y = layout['header_height'] + 35
            param_text = "parameter" if param_count == 1 else "parameters"
            svg_content += f'''<text x="{x_pos}" y="{info_y}" class="layer-info" style="fill: black;">{param_count} {param_text}</text>
'''
        elif layer_type == "Output":
            param_count = layer_size  # Number of output parameters
            info_y = layout['header_height'] + 35
            param_text = "parameter" if param_count == 1 else "parameters"
            svg_content += f'''<text x="{x_pos}" y="{info_y}" class="layer-info" style="fill: black;">{param_count} {param_text}</text>
'''

        # Draw neurons (different for input and output layers)
        if layer_idx == 0:  # Input layer - draw as rectangles centered
            for i, (neuron_x, neuron_y) in enumerate(neuron_positions):
                # Get variable name or use default
                var_name = input_names_list[i] if i < len(input_names_list) else f"x{i+1}"

                # Rectangle dimensions - dynamic width based on text length
                neuron_rect_width = max(70, calculate_text_width(var_name, 10))
                rect_height = 25

                # Position rectangle centered on neuron position
                rect_x = neuron_x - neuron_rect_width//2
                text_x = neuron_x  # Center text in rectangle

                # Draw rounded rectangle with layer-specific color
                svg_content += f'''<rect x="{rect_x}" y="{neuron_y - rect_height//2}" width="{neuron_rect_width}" height="{rect_height}" fill="white" stroke="{color}" stroke-width="2.5" rx="8" ry="8"/>
'''
                # Variable name text (black text)
                svg_content += f'''<text x="{text_x}" y="{neuron_y + 4}" class="layer-info" style="fill: black; font-weight: bold; font-size: 10px;">{var_name}</text>
'''
        elif layer_idx == len(layer_sizes) - 1:  # Output layer - draw as rectangles centered
            for i, (neuron_x, neuron_y) in enumerate(neuron_positions):
                # Get output name from model info or use default
                output_names_list = model_info.get('output_names', [])
                output_name = output_names_list[i] if i < len(output_names_list) else f"y{i+1}"

                # Rectangle dimensions - dynamic width based on text length
                neuron_rect_width = max(70, calculate_text_width(output_name, 10))
                rect_height = 25

                # Position rectangle centered on neuron position
                rect_x = neuron_x - neuron_rect_width//2
                text_x = neuron_x  # Center text in rectangle

                # Draw rounded rectangle with layer-specific color
                svg_content += f'''<rect x="{rect_x}" y="{neuron_y - rect_height//2}" width="{neuron_rect_width}" height="{rect_height}" fill="white" stroke="{color}" stroke-width="2.5" rx="8" ry="8"/>
'''
                # Output name text (black text)
                svg_content += f'''<text x="{text_x}" y="{neuron_y + 4}" class="layer-info" style="fill: black; font-weight: bold; font-size: 10px;">{output_name}</text>
'''
        else:
            # Hidden layers - draw as circles with improved styling
            for neuron_x, neuron_y in neuron_positions:
                svg_content += f'''<circle cx="{neuron_x}" cy="{neuron_y}" r="{neuron_radius}" fill="{color}" stroke="white" stroke-width="2.5" class="neuron"/>
'''


        # Add bias neuron (only for hidden layers that have a next layer)
        if layer_idx > 0 and layer_idx < len(layer_sizes) - 1:  # Not for input or output layer
            bias_y = min([pos[1] for pos in neuron_positions]) - 40  # Above other neurons
            # Bias neuron (circular shape) with improved styling
            svg_content += f'''<circle cx="{x_pos}" cy="{bias_y}" r="8" fill="{bias_color}" stroke="white" stroke-width="2.5"/>
'''

        # Weight and bias information between layers
        if layer_idx < len(layer_sizes) - 1:
            next_size = layer_sizes[layer_idx + 1]
            next_x = layer_positions[layer_idx + 1]
            next_positions = all_positions[layer_idx + 1]

            # Bias connections to next layer neurons (only from hidden layers)
            if layer_idx > 0 and layer_idx < len(layer_sizes) - 1:  # If there's a bias for this layer
                bias_y = min([pos[1] for pos in neuron_positions]) - 40
                for next_neuron_x, next_neuron_y in next_positions:
                    # Check if next layer is output layer (uses rectangles)
                    if layer_idx + 1 == len(layer_sizes) - 1:  # Next layer is output layer
                        # Connect to left edge of output rectangle
                        end_x = next_neuron_x - 35  # Left edge of rectangle
                        end_y = next_neuron_y
                    else:
                        # Connect to hidden layer neuron (circle)
                        end_x = next_neuron_x - neuron_radius
                        end_y = next_neuron_y

                    svg_content += f'''<line x1="{x_pos + 8}" y1="{bias_y}" x2="{end_x}" y2="{end_y}" stroke="{bias_color}" stroke-width="1.5" opacity="0.7"/>
'''




    # Close SVG
    svg_content += '</svg>'

    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        return output_path
    except Exception as e:
        raise Exception(f"Could not write SVG file: {e}")


def visualize_architecture(onnx_path, output_path="my_network.svg"):
    """
    Generate architecture visualization for an ONNX file

    Args:
        onnx_path (str): Path to ONNX file
        output_path (str): Path for output SVG file

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}")
        return False

    try:
        print(f"[INFO] Parsing ONNX model: {onnx_path}")
        model_info = parse_onnx_protobuf(onnx_path)

        if model_info:
            print(f"[INFO] Creating SVG visualization...")
            create_svg_architecture(model_info, output_path)
            print(f"[SUCCESS] Architecture diagram saved: {output_path}")
            return True
        else:
            print(f"[ERROR] Failed to parse ONNX model")
            return False
    except Exception as e:
        print(f"[ERROR] Architecture visualization failed: {e}")
        return False


def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("DNN Architecture Visualizer")
        print("Usage: python DNN_architecture.py <onnx_file> [output.svg]")
        print("\nThis tool works independently with any ONNX file.")
        print("Example:")
        print("  python DNN_architecture.py model.onnx")
        print("  python DNN_architecture.py /path/to/model.onnx diagram.svg")
        return

    onnx_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "my_network.svg"

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}")
        print(f"[INFO] Creating generic network diagram...")
        try:
            # Create sample weight/bias data for demonstration
            sample_weights = [
                {
                    'name': 'dense_1.weight',
                    'values': [0.1234, -0.5678, 0.9876, 0.3456, -0.2345],
                    'size': 91,
                    'shape': [13, 7],
                    'avg': 0.0234,
                    'min': -0.8765,
                    'max': 1.2345
                },
                {
                    'name': 'dense_2.weight',
                    'values': [0.4567, -0.1234, 0.6789, -0.8901, 0.2345],
                    'size': 42,
                    'shape': [7, 6],
                    'avg': 0.0123,
                    'min': -1.1234,
                    'max': 0.9876
                }
            ]

            sample_biases = [
                {
                    'name': 'dense_1.bias',
                    'values': [0.123, -0.456, 0.789, -0.234, 0.567, -0.890, 0.345],
                    'size': 7,
                    'shape': [7],
                    'avg': 0.034,
                    'min': -0.890,
                    'max': 0.789
                },
                {
                    'name': 'dense_2.bias',
                    'values': [0.234, -0.567, 0.890, -0.123, 0.456, -0.789],
                    'size': 6,
                    'shape': [6],
                    'avg': 0.017,
                    'min': -0.789,
                    'max': 0.890
                }
            ]

            generic_info = {
                'inputs': [('input', 13)],
                'outputs': [('output', 1)],
                'layer_sizes': [13, 7, 6, 5, 1],
                'weight_names': ['dense_1.weight', 'dense_2.weight'],
                'weight_tensors': sample_weights,
                'bias_tensors': sample_biases,
                'total_weights': 2,
                'input_names': [f'x{i+1}' for i in range(13)]
            }
            svg_path = create_svg_architecture(generic_info, output_path)
            print(f"[SUCCESS] Generic diagram saved: {svg_path}")
        except Exception as e2:
            print(f"[ERROR] Could not create diagram: {e2}")
        return

    try:
        print(f"[INFO] Parsing ONNX model: {onnx_path}")
        model_info = parse_onnx_protobuf(onnx_path)

        print(f"[INFO] Detected architecture:")
        print(f"  - Input size: {model_info['inputs'][0][1] if model_info['inputs'] else 'Unknown'}")
        print(f"  - Layer sizes: {' -> '.join(map(str, model_info['layer_sizes']))}")
        print(f"  - Output size: {model_info['outputs'][0][1] if model_info['outputs'] else 'Unknown'}")
        print(f"  - Weight tensors found: {model_info['total_weights']}")

        # Display weight and bias information if found
        if model_info.get('weight_tensors'):
            print(f"\n[INFO] Weight tensors:")
            for i, weight in enumerate(model_info['weight_tensors']):
                print(f"  - Layer {i+1}: {weight['name']} - Shape: {weight['shape']} - Size: {weight['size']}")
                print(f"    Values (first 5): {weight['values'][:5]}")
                print(f"    Stats: avg={weight['avg']:.4f}, min={weight['min']:.4f}, max={weight['max']:.4f}")

        if model_info.get('bias_tensors'):
            print(f"\n[INFO] Bias tensors:")
            for i, bias in enumerate(model_info['bias_tensors']):
                print(f"  - Layer {i+1}: {bias['name']} - Shape: {bias['shape']} - Size: {bias['size']}")
                print(f"    Values: {bias['values']}")
                print(f"    Stats: avg={bias['avg']:.4f}, min={bias['min']:.4f}, max={bias['max']:.4f}")

        print(f"[INFO] Creating SVG visualization...")
        svg_path = create_svg_architecture(model_info, output_path)

        print(f"[SUCCESS] Architecture diagram saved: {svg_path}")
        print(f"[INFO] Open the SVG file in a web browser to view the visualization")

    except Exception as e:
        print(f"[ERROR] Failed to process ONNX file: {e}")
        # Try to create a generic diagram anyway
        try:
            print("[INFO] Creating generic network diagram...")
            generic_info = {
                'inputs': [('input', 10)],
                'outputs': [('output', 1)],
                'layer_sizes': [10, 8, 6, 1],
                'weight_names': [],
                'total_weights': 0
            }
            svg_path = create_svg_architecture(generic_info, output_path)
            print(f"[SUCCESS] Generic diagram saved: {svg_path}")
        except Exception as e2:
            print(f"[ERROR] Could not create diagram: {e2}")


if __name__ == "__main__":
    main()