import os
import onnxruntime as ort
import onnx
import numpy as np

def check(file_path, verbose=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError("Error: ONNX file does not exist in the folder.")
    if verbose == True:
        print("ONNX ONNX found successfully.\n")

def read(file_path):
    return onnx.load(file_path), ort.InferenceSession(file_path)

def get_input_output_variables(model, session):

    metadata = {prop.key: prop.value for prop in model.metadata_props}
    
    def get_real_name(name, prefix):
        for key, value in metadata.items():
            if key.startswith(prefix) and key.endswith("_" + str(name)):
                return value
        return name  # Ritorna il nome originale se non trova corrispondenza
    
    input_shape = [inp.shape for inp in session.get_inputs()][0][1]
    input_names = [get_real_name(name, "AMESIM_NN_NAME_INPUT_") for name in range(1,input_shape+1)]

    output_shape = [out.shape for out in session.get_outputs()][0][1]
    output_names = [get_real_name(name, "AMESIM_NN_NAME_OUTPUT_") for name in range(1,output_shape+1)]

    return input_names, output_names, input_shape, output_shape

def infere(input_data, session):
    """ 
    Infers the ONNX model with the given input data

    input_data: numpy array of shape (n,)
    session: ONNX runtime session 
    """
    
    input_name = session.get_inputs()[0].name  # Assume un solo input

    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)  # Aggiunge batch dimension
    
    input_feed = {input_name: input_data.astype(np.float32)}

    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, input_feed)

    return outputs[0][0]

def initialize(file_path, verbose=False):
    """ Initializes the ONNX model and returns input/output names and shapes """

    check(file_path, verbose)
    model, session = read(file_path)
    
    input_names, output_names, input_shape, output_shape = get_input_output_variables(model, session)
        
    if verbose == True:
        print("ONNX caricato con successo.\n")
        print("Variabili di input:", input_names)
        print("Variabili di output:", output_names, "\n")
    
    # Restituisci tutto ciò che serve per le simulazioni future
    return input_names, output_names, input_shape, output_shape

if __name__ == "__main__":
    # Example usage
    file_path = "DNN.onnx"
    input_names, output_names, input_shape, output_shape = initialize(file_path, verbose=False)

    # Example input data
    input_data = [16,	0.086734694,	0.036283917,	0.008571429,	-0.27755102,	-0.07755102,	1.363265306,	7040.816327]

    output = infere(np.array(input_data), ort.InferenceSession(file_path))
    print("Output:", output)

  