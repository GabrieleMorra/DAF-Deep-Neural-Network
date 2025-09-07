#!/usr/bin/env python3
"""
DAF Neural Network - Single Model Training
Console-based training script for single neural network models
"""

from daf_neural_network.core.trainer import TrainNeuralNetwork
from daf_neural_network.visualization.scientific_plots import visualize_NN_results
from daf_neural_network.data.preprocessing import get_database
from daf_neural_network.utils.config import load_config, convert_json_format
from daf_neural_network.utils.helpers import ensure_output_directory

import pickle
import os

def main():
    """Main entry point for single model training"""
    # Ensure output directories exist
    ensure_output_directory()
    
    # Load configuration
    config_file = "neural_network.json"
    try:
        read_json = load_config(config_file)
    except FileNotFoundError:
        print(f"[ERROR] Configuration file '{config_file}' not found.")
        print("Expected locations: configs/neural_network.json or ./neural_network.json")
        return
    
    # Convert to legacy format for compatibility
    nnModel = convert_json_format(read_json)

    # Load and preprocess database
    try:
        Database = get_database(nnModel)
        print(f"[SUCCESS] Successfully loaded dataset: {nnModel['NeuralNetworkModel']['InputFileName']}")
    except Exception as e:
        print(f"[ERROR] Error loading database: {e}")
        return

    # Extract dataset name for automatic naming
    input_filename = nnModel["NeuralNetworkModel"]["InputFileName"]
    dataset_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_file = f"data/output/Trained_DNN_{dataset_name}"

    print("[INFO] Starting neural network training...")
    print(f"[INFO] Dataset: {len(Database['X_train'])} training samples, {len(Database['X_valid'])} validation samples")
    print(f"[INFO] Target: {len(nnModel['NeuralNetworkModel']['outputEntryIndices'])} output variables")
    
    # Train the neural network
    try:
        result = TrainNeuralNetwork(nnModel, Database, silent_mode=False)
        
        if result is not None:
            print(f"[SUCCESS] Training completed successfully!")
            session_dir = result.get('session_dir', 'data/output')
            print(f"[INFO] Model saved in: {session_dir}/")
            
            print("\n[INFO] Generating comprehensive visualizations...")
            viz_dir = f"{session_dir}/visualizations"
            visualize_NN_results(output_dir=viz_dir)
            print(f"[SUCCESS] Visualizations saved in {viz_dir}/")
        else:
            print("[ERROR] Training failed or was interrupted")
            
    except Exception as e:
        print(f"[ERROR] Training error: {e}")
        # Try to load existing model if training failed
        try:
            with open(f"{output_file}.pkl", "rb") as f:
                result = pickle.load(f)
                print(f"[SUCCESS] Loaded existing trained model from '{output_file}.pkl'")
                
                print("\n[INFO] Generating visualizations from existing model...")
                visualize_NN_results(output_dir="data/output/visualizations")
                print("[SUCCESS] Visualizations completed")
        except FileNotFoundError:
            print(f"[ERROR] No existing model found at '{output_file}.pkl'")
            print("Please check the configuration and try again.")
        except Exception as load_error:
            print(f"[ERROR] Error loading existing model: {load_error}")

if __name__ == "__main__":
    main()