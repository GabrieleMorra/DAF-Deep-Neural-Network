#!/usr/bin/env python3
"""
DAF Neural Network - Architecture Sweep
GUI-based script for systematic architecture exploration and optimization
"""

import json
import threading
import multiprocessing as mp
from itertools import product
import os
import time
import numpy as np
from collections import defaultdict
import threading
import time
from daf_neural_network.gui.realtime_monitor import RealTimeTableGUI
from daf_neural_network.data.preprocessing import get_database
from daf_neural_network.core.trainer import TrainNeuralNetwork
from daf_neural_network.utils.config import load_config, convert_json_format
from daf_neural_network.utils.helpers import ensure_output_directory
from daf_neural_network.utils.process_manager import ProcessPoolManager, cleanup_child_processes, setup_process_affinity, create_pause_check_function
from multiprocessing import Queue, Manager
from queue import Empty
import pickle
import psutil
import gc
import datetime
import csv
import subprocess
import platform

def main():
    """Main entry point for architecture sweep"""
    # Ensure output directories exist
    ensure_output_directory()
    
    print("DAF Neural Network - Architecture Sweep")
    print("=" * 50)
    
    # Load sweep configuration
    config_file = "architecture_sweep.json"
    try:
        config = load_config(config_file)
        print(f"[SUCCESS] Configuration loaded from: {config_file}")
    except FileNotFoundError:
        print(f"[ERROR] Configuration file '{config_file}' not found.")
        print("Expected locations: configs/architecture_sweep.json or ./architecture_sweep.json")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        return

    sweep_config = config["SweepConfiguration"]
    nn_model_base = config["NeuralNetworkModel"]

    # Load and preprocess database once for all architectures
    try:
        # Create a temporary neural network config to load the database
        temp_nn = {"NeuralNetworkModel": nn_model_base}
        database = get_database(temp_nn)
        print(f"[SUCCESS] Dataset loaded: {nn_model_base['InputFileName']}")
        print(f"[INFO] Training samples: {len(database['X_train'])}, Validation samples: {len(database['X_valid'])}")
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found: {nn_model_base['InputFileName']}")
        return
    except Exception as e:
        print(f"[ERROR] Error loading database: {e}")
        return

    # Generate all architecture combinations
    architectures = generate_architectures(sweep_config, nn_model_base)
    print(f"[INFO] Generated {len(architectures)} unique architectures to test")
    
    max_threads = sweep_config.get("max_threads", min(4, mp.cpu_count()))
    print(f"[INFO] Using {max_threads} parallel processes")
    print(f"[INFO] Available CPU cores: {mp.cpu_count()}")
    print(f"[INFO] CPU utilization will be distributed across cores")

    # Create multiprocessing manager for shared data structures
    manager = Manager()
    
    # Create data queue for real-time GUI updates (multiprocessing-safe)
    data_queue = manager.Queue()
    
    # Generate parameter combinations for GUI
    param_combinations = generate_param_combinations(sweep_config)
    
    # Pause/delete state shared between GUI and training processes
    pause_state = manager.dict()
    pause_state["paused"] = manager.list()
    pause_state["deleted"] = manager.list()
    
    # Queue for new configurations to be added dynamically
    new_config_queue = manager.Queue()
    
    # Initialize GUI with pause state and new config queue
    print("[GUI] Starting real-time monitoring GUI...")
    gui = RealTimeTableGUI(param_combinations, max_epochs=nn_model_base.get('epochs', 20000), config_path="configs/architecture_sweep.json", data_queue=data_queue, pause_state=pause_state, new_config_queue=new_config_queue, database_info=database)
    
    # Prepare architecture training jobs
    jobs = []
    for i, arch_config in enumerate(architectures):
        model_id = get_architecture_id(arch_config)
        jobs.append((i, model_id, arch_config, database))

    print(f"\n[INFO] Starting parallel training of {len(jobs)} architectures...")
    
    # Training state shared between threads
    training_state = {"results": [], "completed": False, "stop_requested": False, "executor": None}
    
    def run_training():
        results = []
        completed_count = 0
        submitted_jobs = set()  # Track which models have been submitted
        
        # Create process pool manager
        process_manager = ProcessPoolManager(max_threads, data_queue, pause_state)
        executor = process_manager.start()
        training_state["executor"] = executor
        training_state["process_manager"] = process_manager

        # Add all jobs to the manager
        process_manager.add_jobs(jobs)

        # Submit initial batch
        if not process_manager.submit_initial_batch(train_single_architecture):
            return
            
        try:
            # Keep loop running until stop is requested
            while not training_state["stop_requested"]:

                # Process dynamic configurations from GUI
                try:
                    while True:
                        config_tuple = new_config_queue.get(block=False)

                        # Convert config back to architecture
                        hidden_neurons = parse_config_tuple(config_tuple)
                        arch_config = create_architecture_config(nn_model_base, hidden_neurons)
                        model_id = get_architecture_id(arch_config)

                        # Add dynamic job
                        job_id = len(submitted_jobs)
                        new_job = (job_id, model_id, arch_config, database)
                        process_manager.add_dynamic_job(new_job, train_single_architecture)
                        submitted_jobs.add(model_id)

                except Empty:
                    pass  # No more configurations in queue

                # Check if stop was requested
                if training_state["stop_requested"]:
                    print("[INFO] Stopping remaining training jobs...")
                    process_manager.request_stop()
                    break

                # Process completed jobs
                completed_results = process_manager.process_completed_jobs(train_single_architecture)
                if completed_results:
                    results.extend(completed_results)
                    completed_count += len(completed_results)

                    # Periodic memory cleanup
                    if completed_count % 5 == 0:
                        gc.collect()

                # Check if all jobs are completed
                if process_manager.is_complete() and new_config_queue.qsize() == 0:
                    print(f"[INFO] All jobs completed! Processed {completed_count} total jobs.")
                    break

                # Brief sleep for responsiveness
                time.sleep(0.1)
        
        finally:
            # Use process manager for clean shutdown
            process_manager.force_shutdown()
        
        training_state["results"] = results
        training_state["completed"] = True
        training_state["executor"] = None
        
        # Only print results if not stopped
        if not training_state["stop_requested"]:
            # Save results summary
            print(f"\n[COMPLETED] Architecture sweep completed!")
            print(f"[SUCCESS] Successfully trained: {len(results)} models")
            print(f"[INFO] Failed: {len(jobs) - len(results)} models")
            
            if results:
                save_sweep_results(results, nn_model_base)
                print("[INFO] Results saved to: data/output/sweep_results.csv")
        else:
            print("[INFO] Training stopped by user.")
    
    # Start training in background thread (NOT daemon so it stays alive)
    training_thread = threading.Thread(target=run_training, daemon=False)
    training_thread.start()
    
    # Start data collection after GUI mainloop starts
    def start_data_collection():
        """Start data collection thread after GUI is ready"""
        data_thread = threading.Thread(target=gui.collect_training_data, args=(data_queue,), daemon=True)
        data_thread.start()
    
    # Schedule data collection to start after GUI is ready
    gui.root.after(100, start_data_collection)
    
    # Set up GUI close handler
    def on_closing():
        """Handle GUI window close with proper process cleanup"""
        print("[INFO] Closing GUI, stopping all processes...")
        training_state["stop_requested"] = True

        # Stop data collection thread
        try:
            if hasattr(gui, 'stop_data_collection'):
                gui.stop_data_collection()
        except Exception as e:
            print(f"[WARNING] Error stopping data collection: {e}")

        # Force shutdown process manager immediately
        if training_state.get("process_manager"):
            process_manager = training_state["process_manager"]
            print("[INFO] Force terminating process pool from GUI close...")
            process_manager.force_shutdown()

        # Brief wait then destroy GUI
        time.sleep(0.5)
        try:
            gui.root.destroy()
        except:
            pass
    
    gui.root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run GUI in main thread - this will block until GUI is closed
    print("\n[GUI] GUI is running. Close the GUI window to exit.")
    try:
        gui.show()  # Run GUI in main thread
    except KeyboardInterrupt:
        training_state["stop_requested"] = True
        print("\n[INFO] Shutting down...")
    
    # Wait for training thread to stop cleanly
    if training_state["stop_requested"]:
        print("[INFO] Waiting for training thread to stop...")
        training_thread.join(timeout=3)  # Wait up to 3 seconds
        if training_thread.is_alive():
            print("[WARNING] Training thread did not stop gracefully, forcing exit")
        else:
            print("[INFO] Training thread stopped successfully")

        # Final cleanup using utility function
        cleanup_child_processes()

        print("[INFO] Shutdown complete.")
        # Force exit to ensure all processes are terminated
        os._exit(0)

def parse_config_tuple(config_tuple):
    """Parse configuration tuple into hidden neurons list"""
    # Handle both old format (n1, n2, n3, num_layers) and new format [n1, n2, n3, ..., 0, num_layers]
    if len(config_tuple) == 4:
        # Old format: (n1, n2, n3, num_layers)
        n1, n2, n3, num_layers = config_tuple
        all_neurons = [n1, n2, n3]
    else:
        # New format: [n1, n2, n3, ..., 0, 0, num_layers] where last element is num_layers
        num_layers = config_tuple[-1]  # Last element is always num_layers
        all_neurons = list(config_tuple[:-1])  # All neurons except last element
    
    # Extract only the neurons for the number of layers specified
    hidden_neurons = all_neurons[:num_layers]
    
    # Remove any zeros (padding) from the end
    while hidden_neurons and hidden_neurons[-1] == 0:
        hidden_neurons.pop()
    
    return hidden_neurons

def get_layer_configs(sweep_config):
    """Extract layer configurations from sweep config"""
    layer_configs = {}
    layer_counts = sweep_config["total_layers_range"]
    
    for i in range(1, max(layer_counts) + 1):
        layer_key = f"{'first' if i == 1 else 'second' if i == 2 else 'third'}_hidden_neurons_range"
        if layer_key in sweep_config:
            layer_configs[i] = sweep_config[layer_key]
    
    return layer_configs, layer_counts

def generate_param_combinations(sweep_config):
    """Generate parameter combinations for GUI initialization"""
    combinations = []
    layer_configs, layer_counts = get_layer_configs(sweep_config)

    for num_layers in layer_counts:
        if num_layers == 1 and 1 in layer_configs:
            for n1 in layer_configs[1]:
                combinations.append((n1, 0, 0, num_layers))
        elif num_layers == 2 and 1 in layer_configs and 2 in layer_configs:
            for n1, n2 in product(layer_configs[1], layer_configs[2]):
                combinations.append((n1, n2, 0, num_layers))
        elif num_layers == 3 and all(i in layer_configs for i in [1, 2, 3]):
            for n1, n2, n3 in product(layer_configs[1], layer_configs[2], layer_configs[3]):
                combinations.append((n1, n2, n3, num_layers))

    return combinations

def generate_architectures(sweep_config, nn_model_base):
    """Generate all possible architecture combinations"""
    architectures = []
    layer_configs, layer_counts = get_layer_configs(sweep_config)
    
    # Check for missing layer configurations
    for i in range(1, max(layer_counts) + 1):
        layer_key = f"{'first' if i == 1 else 'second' if i == 2 else 'third'}_hidden_neurons_range"
        if i not in layer_configs:
            print(f"[WARNING] {layer_key} not found in config, skipping layers > {i-1}")
            break

    for num_layers in layer_counts:
        if num_layers == 1 and 1 in layer_configs:
            for n1 in layer_configs[1]:
                arch = create_architecture_config(nn_model_base, [n1])
                architectures.append(arch)
        elif num_layers == 2 and 1 in layer_configs and 2 in layer_configs:
            for n1, n2 in product(layer_configs[1], layer_configs[2]):
                arch = create_architecture_config(nn_model_base, [n1, n2])
                architectures.append(arch)
        elif num_layers == 3 and all(i in layer_configs for i in [1, 2, 3]):
            for n1, n2, n3 in product(layer_configs[1], layer_configs[2], layer_configs[3]):
                arch = create_architecture_config(nn_model_base, [n1, n2, n3])
                architectures.append(arch)

    return architectures

def create_architecture_config(base_config, hidden_neurons):
    """Create architecture configuration"""
    config = {"NeuralNetworkModel": base_config.copy()}
    
    input_dim = len(base_config["inputEntryIndices"])
    output_dim = len(base_config["outputEntryIndices"])
    
    for i, neurons in enumerate(hidden_neurons, 1):
        prev_dim = input_dim if i == 1 else hidden_neurons[i-2]
        config[f"Layer{i}"] = {
            "input_dim": prev_dim,
            "output_dim": neurons,
            "activation": "elu"  # Always use elu for hidden layers
        }
    
    # Output layer - use the same activation as the base model
    final_layer = len(hidden_neurons) + 1
    prev_dim = hidden_neurons[-1]
    config[f"Layer{final_layer}"] = {
        "input_dim": prev_dim,
        "output_dim": output_dim,
        "activation": "tanh"  # Use tanh as in the original config
    }
    
    return config

def get_architecture_id(config):
    """Generate unique ID for architecture"""
    layers = [key for key in config.keys() if key.startswith("Layer") and key != f"Layer{len([k for k in config.keys() if k.startswith('Layer')])}"]
    neurons = [str(config[layer]["output_dim"]) for layer in sorted(layers)]
    return "x".join(neurons)

def train_single_architecture(job_id, model_id, arch_config, database, data_queue, pause_state=None):
    """Train a single architecture with dedicated CPU core assignment"""
    try:
        # Set up CPU affinity and process priority
        setup_process_affinity(job_id)

        # Create pause check function
        pause_check = create_pause_check_function(pause_state, model_id)
        
        # Train the model with pause check
        result = TrainNeuralNetwork(
            arch_config, 
            database, 
            model_id=model_id,
            data_queue=data_queue,
            silent_mode=True,
            pause_check_func=pause_check
        )
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Error training {model_id}: {e}")
        return None

def save_sweep_results(results, base_config):
    """Save architecture sweep results to CSV"""
    os.makedirs("data/output", exist_ok=True)
    
    with open("data/output/sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Architecture", "Training_Fidelity", "Validation_Fidelity", 
            "Mean_Loss", "Elapsed_Time", "R2_Scores"
        ])
        
        for model_id, result in results:
            writer.writerow([
                model_id,
                f"{result['training_fidelity']:.3f}",
                f"{result['validation_fidelity']:.3f}",
                f"{result['mean_loss']:.6e}",
                result['elapsed_time'],
                ";".join(f"{score:.3f}" for score in result['r2_per_variable'])
            ])

if __name__ == "__main__":
    main()