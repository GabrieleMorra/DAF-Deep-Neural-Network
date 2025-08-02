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
from concurrent.futures import ThreadPoolExecutor, as_completed
from realtime_gui import RealTimeTableGUI
from GetDatabase import get_database
from Train import TrainNeuralNetwork
from queue import Queue
import pickle

def run_model_threaded(first_hidden_neurons, second_hidden_neurons, total_layers, database, sweep_config, results_queue, data_queue, onnx_dir="trained_onnx_models"):
    # total_layers: 2 = Input + FirstHidden + Output, 3 = Input + FirstHidden + SecondHidden + Output
    hidden_layers = total_layers - 1  # Subtract output layer
    model_id = f"{first_hidden_neurons}_{second_hidden_neurons}_{total_layers}L"
    if not sweep_config["NeuralNetworkModel"].get("silent_mode", False):
        print(f"[TRAINING] Starting model configuration: {model_id}")
    
    # Costruisci la configurazione dal template
    config = {
        "NeuralNetworkModel": sweep_config["NeuralNetworkModel"].copy()
    }
    
    # Set output file path in ONNX directory
    config["NeuralNetworkModel"]["OutputFileName"] = os.path.join(onnx_dir, f"Trained_DNN_{model_id}")
    
    # Note: InputLayer is not defined in LayerTemplate anymore, it's handled automatically by the network
    # We only define the hidden layers and output layer
    
    # Add hidden layers based on total_layers
    # total_layers = 2: Input + FirstHidden + Output
    # total_layers = 3: Input + FirstHidden + SecondHidden + Output
    config["HiddenLayer1"] = {
        "neurons": first_hidden_neurons, 
        "activation": sweep_config["LayerTemplate"]["FirstHiddenLayer"]["activation"]
    }
    
    if hidden_layers > 1:  # Add second hidden layer for 3-layer networks
        config["HiddenLayer2"] = {
            "neurons": second_hidden_neurons, 
            "activation": sweep_config["LayerTemplate"]["SecondHiddenLayer"]["activation"]
        }
    
    config["OutputLayer"] = {
        "activation": sweep_config["LayerTemplate"]["OutputLayer"]["activation"]
    }
    
    # Converti al formato richiesto da TrainNeuralNetwork
    nnModel = convert_json_format(config)
    
    try:
        # Chiama direttamente TrainNeuralNetwork invece di subprocess
        result = TrainNeuralNetwork(nnModel, database, model_id, data_queue)
        
        # Send completion signal to GUI
        data_queue.put((model_id, 'COMPLETED'))
        
        results_queue.put((model_id, True, result))
        if not sweep_config["NeuralNetworkModel"].get("silent_mode", False):
            print(f"[COMPLETED] Model {model_id} - Validation R²: {result['validation_fidelity']:.2f}%")
        return True
        
    except Exception as e:
        # Send error signal to GUI
        data_queue.put((model_id, 'ERROR'))
        
        if not sweep_config["NeuralNetworkModel"].get("silent_mode", False):
            print(f"[ERROR] Model {model_id} failed: {str(e)}")
        results_queue.put((model_id, False, None))
        return False

class RealTimeGUIWrapper:
    def __init__(self, param_combinations, max_epochs=15000):
        self.param_combinations = param_combinations
        self.max_epochs = max_epochs
        self.gui = None
        self.gui_thread = None
        self.data_queue = Queue()
        self.executor = None  # Store reference to ThreadPoolExecutor
        
    def start_gui(self):
        """Start the GUI in a separate thread"""
        def run_gui():
            self.gui = RealTimeTableGUI(self.param_combinations, self.max_epochs, "NeuralNetworkSweep.json")
            # Start data collection in GUI
            data_thread = threading.Thread(target=self.gui.collect_training_data, args=(self.data_queue,))
            data_thread.daemon = True
            data_thread.start()
            
            self.gui.show()
            
        self.gui_thread = threading.Thread(target=run_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()
        
        # Give GUI time to initialize
        time.sleep(2)
        
    def collect_training_data(self, data_queue):
        """Collect data from training threads and forward to GUI"""
        while True:
            try:
                data = data_queue.get(timeout=1)
                if data is None:
                    self.data_queue.put(None)  # Signal GUI to stop
                    break
                    
                # Forward data to GUI
                self.data_queue.put(data)
                
            except:
                continue
                
    def check_and_update_plot(self):
        """Compatibility method - GUI updates automatically"""
        return True
        
    def close_plot(self):
        """Close the GUI safely"""
        if self.gui:
            self.gui.close()
            # Give GUI thread time to close properly
            time.sleep(1)
    
    def set_executor(self, executor):
        """Set reference to ThreadPoolExecutor for cleanup"""
        self.executor = executor
                
    def log_final_results(self, results):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.log_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, 'SWEEP_SUMMARY', '', '', 'sweep_start'])
            
            for model_id, success, result in results:
                if success and result:
                    writer.writerow([
                        timestamp, 
                        model_id, 
                        'FINAL', 
                        f"{result['validation_fidelity']:.4f}", 
                        'final_result'
                    ])
                    writer.writerow([
                        timestamp, 
                        model_id, 
                        'TRAINING_FIDELITY', 
                        f"{result['training_fidelity']:.4f}", 
                        'final_result'
                    ])
                    writer.writerow([
                        timestamp, 
                        model_id, 
                        'LOSS', 
                        f"{result['mean_loss']:.6e}", 
                        'final_result'
                    ])
                    writer.writerow([
                        timestamp, 
                        model_id, 
                        'TIME', 
                        result['elapsed_time'], 
                        'final_result'
                    ])
                else:
                    writer.writerow([timestamp, model_id, 'FAILED', '0', 'final_result'])
            
            writer.writerow([timestamp, 'SWEEP_SUMMARY', '', '', 'sweep_end'])

def run_sweep_threading():
    # Carica la configurazione dello sweep
    with open("NeuralNetworkSweep.json", "r") as f:
        sweep_config = json.load(f)
    
    first_hidden_neurons_range = sweep_config["SweepConfiguration"]["first_hidden_neurons_range"]
    second_hidden_neurons_range = sweep_config["SweepConfiguration"]["second_hidden_neurons_range"]
    total_layers_range = sweep_config["SweepConfiguration"]["total_layers_range"]
    max_threads = sweep_config["SweepConfiguration"]["max_threads"]
    print("\n" + "="*80)
    print("           NEURAL NETWORK ARCHITECTURE SWEEP")
    print("="*80)
    
    # Load database once using a temporary model configuration
    print("\n[INIT] Loading dataset and initializing base configuration...")
    # Crea un modello temporaneo per caricare il database
    temp_config = {
        "NeuralNetworkModel": sweep_config["NeuralNetworkModel"],
        "HiddenLayer1": {"neurons": 10, "activation": "elu"},
        "OutputLayer": {"activation": "tanh"}
    }
    base_nn = convert_json_format(temp_config)
    database = get_database(base_nn)
    
    # Crea tutte le combinazioni di parametri
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
    
    print(f"[FILTER] Original combinations: {len(all_combinations)}")
    print(f"[FILTER] Unique architectures: {len(param_combinations)}")
    print(f"[FILTER] Eliminated {len(all_combinations) - len(param_combinations)} duplicate architectures")
    
    print(f"[SWEEP] Starting sweep with {len(param_combinations)} model configurations")
    print(f"[CONFIG] Max threads: {max_threads}")
    print(f"[CONFIG] First hidden neurons range: {first_hidden_neurons_range}")
    print(f"[CONFIG] Second hidden neurons range: {second_hidden_neurons_range}")
    print(f"[CONFIG] Total layers range: {total_layers_range} (Input + Hidden + Output)")
    
    # Initialize communication queues
    results_queue = Queue()
    data_queue = Queue()
    
    # Initialize real-time plotter with all configurations
    print("\n[PLOT] Initializing real-time performance monitor...")
    
    # Get max epochs from configuration
    max_epochs = sweep_config["NeuralNetworkModel"]["epochs"]
    
    # Create ONNX models directory
    onnx_dir = "trained_onnx_models"
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
        print(f"[INIT] Created directory for ONNX models: {onnx_dir}")
    
    # Initialize GUI-based real-time plotter
    print("[GUI] Starting real-time GUI monitor...")
    plotter = RealTimeGUIWrapper(param_combinations, max_epochs)
    plotter.start_gui()
    
    # Start data collection thread
    data_thread = threading.Thread(target=plotter.collect_training_data, args=(data_queue,))
    data_thread.daemon = True
    data_thread.start()
    
    print("[GUI] Real-time GUI monitor activated - Updates every 250 epochs")
    print("[GUI] GUI window should now be visible with real-time plot")
    
    start_time = time.time()
    
    # Implement thread pool that starts new training as soon as a thread finishes
    
    print(f"[THREADS] Starting thread pool with {max_threads} concurrent workers")
    print(f"[THREADS] This will keep {max_threads} training tasks running simultaneously")
    print(f"[THREADS] As each task completes, a new one will start immediately")
    
    # Use ThreadPoolExecutor for better thread management
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Give GUI reference to executor for cleanup
        plotter.set_executor(executor)
        # Submit all tasks to the thread pool
        future_to_config = {}
        
        print(f"[POOL] Submitting {len(param_combinations)} training tasks to thread pool...")
        
        for i, (first_hidden_neurons, second_hidden_neurons, total_layers) in enumerate(param_combinations):
            future = executor.submit(
                run_model_threaded, 
                first_hidden_neurons, second_hidden_neurons, total_layers, 
                database, sweep_config, results_queue, data_queue, onnx_dir
            )
            future_to_config[future] = (first_hidden_neurons, second_hidden_neurons, total_layers)
            
            # Log when first batch starts
            if i < max_threads:
                model_id = f"{first_hidden_neurons}_{second_hidden_neurons}_{total_layers}L"
                print(f"[START] Model {model_id} starting in thread pool")
        
        # Process completed tasks as they finish
        completed_count = 0
        total_count = len(param_combinations)
        
        print(f"[POOL] All {total_count} training tasks submitted to thread pool")
        print(f"[POOL] First {min(max_threads, total_count)} tasks are now running")
        
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            first_h, second_h, layers = config
            model_id = f"{first_h}_{second_h}_{layers}L"
            
            try:
                result = future.result()
                completed_count += 1
                remaining = total_count - completed_count
                print(f"[PROGRESS] Completed {completed_count}/{total_count} - Model {model_id} finished")
                if remaining > 0:
                    print(f"[POOL] {remaining} tasks remaining in queue, next task starting automatically")
                
            except Exception as e:
                completed_count += 1
                remaining = total_count - completed_count
                print(f"[ERROR] Model {model_id} failed with exception: {str(e)}")
                if remaining > 0:
                    print(f"[POOL] {remaining} tasks remaining in queue, next task starting automatically")
            
            # Update plot periodically (every 5 completions or at the end)
            if completed_count % 5 == 0 or completed_count == total_count:
                plotter.check_and_update_plot()
        
        print(f"[POOL] All {total_count} training tasks completed")
    
    # Terminate data collection
    data_queue.put(None)
    data_thread.join()
    
    # Final plot update
    plotter.check_and_update_plot()
    
    end_time = time.time()
    
    # Raccogli i risultati
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    total_configurations = len(param_combinations)  # Use the actual number of configurations
    
    print(f"\n[SUMMARY] Models completed successfully: {successful}/{total_configurations}")
    print(f"[SUMMARY] Total execution time: {end_time - start_time:.2f} seconds")
    if total_configurations > 0:
        print(f"[SUMMARY] Average time per model: {(end_time - start_time)/total_configurations:.2f} seconds")
    else:
        print("[SUMMARY] No models were processed")
    
    print("\n" + "="*80)
    print("                          SWEEP COMPLETED")
    print("="*80)
    
    # Show failed configurations
    failed_configs = [model_id for model_id, success, _ in results if not success]
    if failed_configs:
        print(f"\n[FAILURES] Failed model configurations:")
        for config in failed_configs:
            print(f"  → {config}")
    
    # Show best performing models
    successful_results = [(model_id, result) for model_id, success, result in results if success and result]
    if successful_results:
        print(f"\n[RANKINGS] Top performing models (by validation R²):")
        successful_results.sort(key=lambda x: x[1]['validation_fidelity'], reverse=True)
        
        print(f"\n{'Rank':<6}{'Configuration':<15}{'Val R²':<10}{'Train R²':<10}{'Loss':<12}{'Time':<10}")
        print("-"*70)
        for i, (model_id, result) in enumerate(successful_results[:5]):
            print(f"{i+1:<6}{model_id:<15}{result['validation_fidelity']:<10.2f}{result['training_fidelity']:<10.2f}{result['mean_loss']:<12.2e}{result['elapsed_time']:<10}")
    
    print(f"\n[MODELS] All trained ONNX models saved in: {onnx_dir}/")
    print(f"[GUI] Real-time monitor will remain open for analysis")
    print(f"[INFO] Close the GUI window manually when finished")
    print("\n" + "="*80)
    
    # Keep plot open for analysis - don't auto-close
    print("[INFO] GUI remains open for analysis. Close manually when done.")

def convert_json_format(new_nn):
    old_nn = {}
    keys = list(new_nn.keys())
    last_layer_key = next(key for key in keys[::-1] if "Layer" in key)

    previous_neurons = len(new_nn["NeuralNetworkModel"]["inputEntryIndices"])

    for key, value in new_nn.items():
        if "Layer" in key:
            if key == last_layer_key:
                old_nn[key] = {
                    "input_dim": previous_neurons,
                    "output_dim": len(new_nn["NeuralNetworkModel"]["outputEntryIndices"]),
                    "activation": value["activation"]
                }
            else:
                old_nn[key] = {
                    "input_dim": previous_neurons,
                    "output_dim": value["neurons"],
                    "activation": value["activation"]
                }
                previous_neurons = value["neurons"]
        else:
            old_nn[key] = value

    return old_nn

if __name__ == "__main__":
    print("Neural Network Architecture Sweep - Real-Time Performance Monitor")
    print("Initializing comprehensive model evaluation...")
    run_sweep_threading()