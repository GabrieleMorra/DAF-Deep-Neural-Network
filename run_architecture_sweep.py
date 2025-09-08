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
from concurrent.futures import ProcessPoolExecutor, as_completed
from daf_neural_network.gui.realtime_monitor import RealTimeTableGUI
from daf_neural_network.data.preprocessing import get_database
from daf_neural_network.core.trainer import TrainNeuralNetwork
from daf_neural_network.utils.config import load_config, convert_json_format
from daf_neural_network.utils.helpers import ensure_output_directory
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
        
        # Create ProcessPoolExecutor for true parallelization (bypasses GIL)
        executor = ProcessPoolExecutor(max_workers=max_threads)
        training_state["executor"] = executor
        
        try:
            # Submit initial jobs with pause state
            future_to_job = {}
            for job_id, model_id, arch_config, db in jobs:
                future = executor.submit(train_single_architecture, job_id, model_id, arch_config, db, data_queue, pause_state)
                future_to_job[future] = (job_id, model_id)
                submitted_jobs.add(model_id)
            
            print(f"[INFO] Process pool created with {max_threads} worker processes")
            print(f"[INFO] Each process will be assigned to a dedicated CPU core")
            print(f"[INFO] Submitted {len(jobs)} initial jobs")
            
            # Keep loop running until stop is requested - processes stay alive
            loop_count = 0
            while not training_state["stop_requested"]:
                loop_count += 1
                
                # Check for new configurations from queue (thread-safe)
                new_configs_processed = 0
                
                try:
                    while True:  # Process all pending configurations
                        config_tuple = new_config_queue.get(block=False)  # Non-blocking get
                        new_configs_processed += 1
                        
                        # Convert config back to architecture
                        hidden_neurons = parse_config_tuple(config_tuple)
                        
                        # Create architecture config
                        arch_config = create_architecture_config(nn_model_base, hidden_neurons)
                        model_id = get_architecture_id(arch_config)
                        
                        # Only submit if not already submitted - threads will pick up the work
                        if model_id not in submitted_jobs:
                            job_id = len(submitted_jobs)  # New job ID
                            future = executor.submit(train_single_architecture, job_id, model_id, arch_config, database, data_queue, pause_state)
                            future_to_job[future] = (job_id, model_id)
                            submitted_jobs.add(model_id)
                            print(f"[INFO] Submitted new training job for {model_id}")
                        else:
                            print(f"[WARNING] Skipping duplicate job {model_id}")
                        
                except Empty:
                    # No more configurations in queue, continue with main loop
                    pass
                
                # Check if stop was requested (GUI closed)
                if training_state["stop_requested"]:
                    print("[INFO] Stopping remaining training jobs...")
                    # Cancel remaining futures
                    for remaining_future in future_to_job:
                        remaining_future.cancel()
                    break
                
                # Process completed jobs (with timeout to allow checking for new configs)
                if future_to_job:
                    completed_futures = []
                    for future in list(future_to_job.keys()):
                        if future.done():
                            completed_futures.append(future)
                    
                    if completed_futures:
                        for future in completed_futures:
                            job_id, model_id = future_to_job.pop(future)
                            completed_count += 1
                            
                            try:
                                result = future.result()
                                if result is not None:
                                    results.append((model_id, result))
                                    print(f"[SUCCESS] [{completed_count}] {model_id} completed successfully")
                                else:
                                    print(f"[ERROR] [{completed_count}] {model_id} failed")
                            except Exception as e:
                                print(f"[ERROR] [{completed_count}] {model_id} error: {e}")
                            
                            # Periodic memory cleanup for completed jobs
                            if completed_count % 5 == 0:  # Every 5 completed jobs
                                gc.collect()
                    
                    # Brief wait to allow new configurations to be checked
                    time.sleep(0.1)
                else:
                    # No active jobs, but threads are still alive waiting for work
                    if training_state["stop_requested"]:
                        break
                    
                    # Check if there are items in queue even when no active jobs
                    if new_config_queue.qsize() > 0:
                        continue  # Don't sleep, process immediately
                    
                    time.sleep(0.5)  # Brief sleep for responsive queue checking
        
        finally:
            # Only shutdown executor when explicitly stopping
            print("[INFO] Shutting down thread pool...")
            executor.shutdown(wait=False)  # Don't wait for running tasks to complete
            print("[INFO] Thread pool shutdown initiated")
        
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
        """Handle GUI window close"""
        print("[INFO] Closing GUI, stopping all threads...")
        training_state["stop_requested"] = True
        
        # Stop data collection thread
        try:
            # Signal GUI to stop data collection
            if hasattr(gui, 'stop_data_collection'):
                gui.stop_data_collection()
        except Exception as e:
            print(f"[WARNING] Error stopping data collection: {e}")
        
        # Brief wait for threads to stop
        time.sleep(0.2)
        gui.root.destroy()
    
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
        print("[INFO] Shutdown complete.")
        # Force exit to ensure all threads are terminated
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

def train_single_architecture(job_id, model_id, arch_config, database, data_queue, pause_state=None, gui_ref=None):
    """Train a single architecture with dedicated CPU core assignment"""
    try:
        # Set CPU affinity using psutil for cross-platform support
        process = psutil.Process()
        available_cpus = list(range(psutil.cpu_count()))
        
        if available_cpus:
            # Assign each process to a dedicated CPU core
            assigned_cpu = available_cpus[job_id % len(available_cpus)]
            try:
                process.cpu_affinity([assigned_cpu])
                print(f"[CPU] Process {model_id} assigned to CPU core {assigned_cpu}")
            except (OSError, AttributeError) as e:
                print(f"[WARNING] Could not set CPU affinity for {model_id}: {e}")
        
        # Set process priority to high for better performance
        try:
            if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                process.nice(-5)  # Unix-like systems
        except (OSError, AttributeError):
            pass  # Continue without priority adjustment
        
        # Create pause check function for multiprocessing
        def pause_check():
            if pause_state is None:
                return None  # No pause control
            
            # Note: GUI reference doesn't work across processes, so skip global pause check
            
            # Check if this model is deleted (convert to list for multiprocessing compatibility)
            if model_id in list(pause_state.get('deleted', [])):
                return True  # True = deleted
            
            # Check if this model is paused (convert to list for multiprocessing compatibility)
            if model_id in list(pause_state.get('paused', [])):
                return False  # False = paused
            
            return None  # None = continue training
        
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