import os

def ensure_output_directory():
    """Ensure output directory exists"""
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/output/visualizations", exist_ok=True)

def get_output_filename(nn):
    """Generate output filename based on dataset name"""
    input_filename = nn["NeuralNetworkModel"]["InputFileName"]
    dataset_name = os.path.splitext(os.path.basename(input_filename))[0]
    return f"Trained_DNN_{dataset_name}"