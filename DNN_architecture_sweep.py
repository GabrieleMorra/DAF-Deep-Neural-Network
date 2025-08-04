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
import psutil
import gc
import datetime
import csv
import subprocess
import platform

# Global shutdown flag for all training threads
shutdown_requested = False
shutdown_lock = threading.Lock()

# Core assignment tracking
core_assignment_counter = 0
core_assignment_lock = threading.Lock()

def set_cpu_affinity():
    """Assign thread to a specific physical core for true parallelization"""
    try:
        # Skip for main thread
        if threading.current_thread().name == 'MainThread':
            return
        
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        
        if physical_cores <= 1:
            print("[CPU] Only 1 physical core detected, CPU affinity not set")
            return
            
        # Sequential assignment to physical cores
        global core_assignment_counter
        with core_assignment_lock:
            core_id = core_assignment_counter % physical_cores
            core_assignment_counter += 1
        
        # Calculate logical cores per physical core
        if logical_cores > physical_cores:
            logical_per_physical = logical_cores // physical_cores
            # Assign thread to all logical cores of this physical core
            cpu_list = [core_id * logical_per_physical + i for i in range(logical_per_physical)]
        else:
            # No hyperthreading, assign to single core
            cpu_list = [core_id]
        
        # Set CPU affinity
        current_process = psutil.Process()
        current_process.cpu_affinity(cpu_list)
        
        thread_name = threading.current_thread().name
        print(f"[CPU] Thread {thread_name} assigned to physical core {core_id} (logical cores: {cpu_list})")
        
        # Verify assignment worked
        try:
            actual_affinity = current_process.cpu_affinity()
            if set(actual_affinity) == set(cpu_list):
                print(f"[CPU] ✓ Thread {thread_name} successfully bound to cores {actual_affinity}")
            else:
                print(f"[CPU] ⚠ Thread {thread_name} assignment mismatch: expected {cpu_list}, got {actual_affinity}")
        except:
            pass
        
    except Exception as e:
        print(f"[CPU] Failed to set CPU affinity: {e}")
        pass  # Don't fail training if CPU affinity fails

def run_model_threaded(layer_neurons, database, sweep_config, results_queue, data_queue, gui_ref=None, onnx_dir="trained_onnx_models"):
    # Set CPU affinity for this thread to distribute across physical cores
    set_cpu_affinity()
    
    # layer_neurons is a list: [n1, n2, n3, ..., 0, 0, total_layers]
    # Extract total_layers (last element) and active neuron counts
    total_layers = layer_neurons[-1]
    active_neurons = layer_neurons[:-1][:total_layers]  # Only take the first total_layers neurons
    
    # Generate appropriate model_id based on architecture
    if total_layers == 1:
        model_id = f"{active_neurons[0]}"  # Single layer: "5", "10", etc.
    else:
        model_id = "x".join(map(str, active_neurons))  # Multiple layers: "5x10", "5x10x15", etc.
    
    if not sweep_config["NeuralNetworkModel"].get("silent_mode", False):
        print(f"[TRAINING] Starting model configuration: {model_id}")
    
    # Costruisci la configurazione dal template
    config = {
        "NeuralNetworkModel": sweep_config["NeuralNetworkModel"].copy()
    }
    
    # Set output file path in ONNX directory
    config["NeuralNetworkModel"]["OutputFileName"] = os.path.join(onnx_dir, f"DNN_{model_id}")
    
    # Note: InputLayer is not defined in LayerTemplate anymore, it's handled automatically by the network
    # We only define the hidden layers and output layer
    
    # Add hidden layers generically based on total_layers
    for i, neurons in enumerate(active_neurons):
        layer_num = i + 1
        # Try to get layer-specific activation, fallback to generic or first layer
        layer_key = f"HiddenLayer{layer_num}"
        template_key = f"HiddenLayer{layer_num}"
        
        # Fallback chain for activation function
        activation = None
        if template_key in sweep_config["LayerTemplate"]:
            activation = sweep_config["LayerTemplate"][template_key]["activation"]
        elif "FirstHiddenLayer" in sweep_config["LayerTemplate"]:
            activation = sweep_config["LayerTemplate"]["FirstHiddenLayer"]["activation"]
        else:
            activation = "elu"  # Default fallback
        
        config[layer_key] = {
            "neurons": neurons,
            "activation": activation
        }
    
    config["OutputLayer"] = {
        "activation": sweep_config["LayerTemplate"]["OutputLayer"]["activation"]
    }
    
    # Converti al formato richiesto da TrainNeuralNetwork
    nnModel = convert_json_format(config)
    
    # Debug: Print actual architecture (always show for debugging)
    layers_info = []
    for key, value in config.items():
        if "Layer" in key and key != "NeuralNetworkModel":
            if "neurons" in value:
                layers_info.append(f"{key}: {value['neurons']} neurons")
            else:
                layers_info.append(f"{key}: output layer")
    print(f"[ARCH DEBUG] {model_id} -> {' | '.join(layers_info)}")
    
    # Also debug the input parameters
    print(f"[PARAMS DEBUG] {model_id} -> neurons: {active_neurons}, total_layers: {total_layers}")
    
    # Check for pause before starting training (global or individual)
    pause_message_shown = False
    
    def check_pause_and_deletion():
        nonlocal pause_message_shown
        
        # Check for global shutdown first
        global shutdown_requested
        import __main__
        if shutdown_requested or (hasattr(__main__, 'shutdown_requested') and __main__.shutdown_requested):
            print(f"[SHUTDOWN] {model_id} terminating due to GUI closure")
            return True  # Should terminate
        
        if gui_ref:
            # Check if model was deleted
            if gui_ref.is_model_deleted(model_id):
                print(f"[DELETED] {model_id} was deleted, terminating training")
                return True  # Should terminate
            
            # Check for pause (global or individual)
            global_paused = gui_ref.is_paused
            individual_paused = gui_ref.is_model_paused(model_id)
            
            if global_paused or individual_paused:
                # Show message only once and update GUI status
                if not pause_message_shown:
                    if global_paused:
                        print(f"[GLOBAL PAUSE] {model_id} paused (global)")
                    elif individual_paused:
                        print(f"[MODEL PAUSE] {model_id} paused (individual)")
                    
                    # Update GUI status to show paused
                    data_queue.put((model_id, 'PAUSED'))
                    pause_message_shown = True
                
                return False  # Should wait but not terminate
            else:
                # If no longer paused, reset message flag and update status
                if pause_message_shown:
                    print(f"[RESUME] {model_id} resumed")
                    # Update GUI status back to training
                    data_queue.put((model_id, 'TRAINING'))
                    pause_message_shown = False
        return None  # Continue normally
    
    # Initial pause check (no debug spam)
    while True:
        check_result = check_pause_and_deletion()
        if check_result is True:  # Deleted
            return False
        elif check_result is False:  # Paused
            time.sleep(1)  # Check every second
            continue
        else:  # Continue
            break
    
    try:
        # Chiama direttamente TrainNeuralNetwork con controllo pause
        result = TrainNeuralNetwork(nnModel, database, model_id, data_queue, gui_ref, check_pause_and_deletion)
        
        # Only send completion if training actually completed (result is not None)
        if result is not None:
            # Send completion signal to GUI with final results
            data_queue.put((model_id, 'COMPLETED', result))
            
            results_queue.put((model_id, True, result))
            print(f"[COMPLETED] Model {model_id} - Validation Fidelity: {result['validation_fidelity']:.2f}%")
        else:
            # Training was terminated/deleted - just mark as failed
            print(f"[TERMINATED] Model {model_id} training was terminated")
            results_queue.put((model_id, False, None))
        
        # Explicit cleanup to free memory
        del nnModel, config
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        # Send error signal to GUI
        data_queue.put((model_id, 'ERROR'))
        
        print(f"[ERROR] Model {model_id} failed: {str(e)}")
        results_queue.put((model_id, False, None))
        
        # Cleanup even on error
        try:
            del nnModel, config
        except:
            pass
        import gc
        gc.collect()
        
        return False
    finally:
        # Final cleanup regardless of outcome
        print(f"[THREAD] Thread for {model_id} cleaning up and releasing resources")

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
        self.gui_thread.daemon = False  # Don't make it a daemon so it keeps program alive
        self.gui_thread.start()
        
        # Give GUI time to initialize
        time.sleep(2)
        
    def collect_training_data(self, data_queue):
        """Collect data from training threads and forward to GUI"""
        from queue import Empty
        while True:
            # Check for shutdown signal
            global shutdown_requested
            import __main__
            if shutdown_requested or (hasattr(__main__, 'shutdown_requested') and __main__.shutdown_requested):
                print("[DATA] Data collection terminating due to shutdown")
                break
                
            try:
                data = data_queue.get(timeout=1)
                if data is None:
                    self.data_queue.put(None)  # Signal GUI to stop
                    break 
                    
                # Forward data to GUI
                self.data_queue.put(data)
                
            except Empty:
                # Empty exceptions are normal when no data is available
                continue
            except Exception as e:
                # Check if it's a GUI-related error (main thread not in main loop)
                if "main thread is not in main loop" in str(e):
                    print("[DATA] GUI main loop ended, stopping data collection")
                    break
                else:
                    print(f"Error in data collection: {e}")
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
    
    def wait_for_gui_close(self):
        """Wait for the GUI thread to finish (when user closes the window)"""
        if self.gui_thread and self.gui_thread.is_alive():
            # Use very short timeout - GUI should close immediately
            self.gui_thread.join(timeout=2.0)
            if self.gui_thread.is_alive():
                print("[WARNING] GUI thread didn't close cleanly, forcing termination")
                # Set global shutdown flag first
                global shutdown_requested
                with shutdown_lock:
                    shutdown_requested = True
                
                # Force terminate the entire process tree
                try:
                    if platform.system() == "Windows":
                        pid = os.getpid()
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                                     timeout=2, capture_output=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    pass
                os._exit(0)
                
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
    
    # Generic approach: support any number of hidden layers
    hidden_layers_config = []
    layer_num = 1
    while f"hidden_layer_{layer_num}_neurons_range" in sweep_config["SweepConfiguration"]:
        hidden_layers_config.append(sweep_config["SweepConfiguration"][f"hidden_layer_{layer_num}_neurons_range"])
        layer_num += 1
    
    # Fallback to old format for backward compatibility
    if not hidden_layers_config:
        hidden_layers_config = [
            sweep_config["SweepConfiguration"]["first_hidden_neurons_range"],
            sweep_config["SweepConfiguration"]["second_hidden_neurons_range"],
            sweep_config["SweepConfiguration"].get("third_hidden_neurons_range", [])
        ]
        # Remove empty configs
        hidden_layers_config = [config for config in hidden_layers_config if config]
    
    total_layers_range = sweep_config["SweepConfiguration"]["total_layers_range"]
    max_hidden_layers = len(hidden_layers_config)
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
    
    # Create unique combinations efficiently for any number of layers
    unique_combinations = []
    
    for num_layers in total_layers_range:
        if num_layers <= max_hidden_layers:
            # Generate all combinations for this number of layers
            if num_layers == 1:
                for neurons in hidden_layers_config[0]:
                    unique_combinations.append([neurons] + [0] * (max_hidden_layers - 1) + [num_layers])
            elif num_layers == 2:
                for n1 in hidden_layers_config[0]:
                    for n2 in hidden_layers_config[1]:
                        unique_combinations.append([n1, n2] + [0] * (max_hidden_layers - 2) + [num_layers])
            elif num_layers == 3:
                for n1 in hidden_layers_config[0]:
                    for n2 in hidden_layers_config[1]:
                        for n3 in hidden_layers_config[2]:
                            unique_combinations.append([n1, n2, n3] + [0] * (max_hidden_layers - 3) + [num_layers])
            # Add more cases as needed, or make it completely dynamic
            else:
                # Dynamic approach for any number of layers
                from itertools import product
                ranges = hidden_layers_config[:num_layers]
                for combination in product(*ranges):
                    config = list(combination) + [0] * (max_hidden_layers - num_layers) + [num_layers]
                    unique_combinations.append(config)
    
    param_combinations = unique_combinations
    
    # Calculate what would have been generated with the old method for comparison
    all_combinations_count = 1
    for layer_config in hidden_layers_config:
        all_combinations_count *= len(layer_config)
    all_combinations_count *= len(total_layers_range)
    
    print(f"[FILTER] Potential combinations (naive approach): {all_combinations_count}")
    print(f"[FILTER] Unique architectures: {len(param_combinations)}")
    print(f"[FILTER] Eliminated {all_combinations_count - len(param_combinations)} duplicate architectures")
    
    print(f"[SWEEP] Starting sweep with {len(param_combinations)} model configurations")
    print(f"[CONFIG] Max threads: {max_threads}")
    for i, layer_config in enumerate(hidden_layers_config):
        print(f"[CONFIG] Hidden layer {i+1} neurons range: {layer_config}")
    print(f"[CONFIG] Total layers range: {total_layers_range} (Hidden + Output)")
    print(f"[CONFIG] Max hidden layers supported: {max_hidden_layers}")
    
    # Initialize communication queues
    results_queue = Queue()
    data_queue = Queue()
    
    # Initialize real-time plotter with all configurations
    print("\n[PLOT] Initializing real-time performance monitor...")
    
    # Get max epochs from configuration
    max_epochs = sweep_config["NeuralNetworkModel"]["epochs"]
    
    # Create ONNX models directory and clean previous models
    onnx_dir = "trained_onnx_models"
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
        print(f"[INIT] Created directory for ONNX models: {onnx_dir}")
    else:
        # Clean existing ONNX models from previous sessions
        import glob
        existing_models = glob.glob(os.path.join(onnx_dir, "*.onnx"))
        if existing_models:
            print(f"[CLEANUP] Removing {len(existing_models)} existing ONNX models from previous sessions...")
            for model_file in existing_models:
                try:
                    os.remove(model_file)
                    print(f"[CLEANUP] Removed: {os.path.basename(model_file)}")
                except Exception as e:
                    print(f"[WARNING] Could not remove {os.path.basename(model_file)}: {e}")
            print(f"[CLEANUP] Directory cleaned for new training session")
        else:
            print(f"[INIT] Using existing directory: {onnx_dir} (no previous models found)")
    
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
    
    # Get CPU information
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    
    print(f"[CPU] Detected {physical_cores} physical cores, {logical_cores} logical cores")
    print(f"[THREADS] Starting thread pool with {max_threads} concurrent workers")
    print(f"[THREADS] Each thread will be assigned to a different physical core for true parallelization")
    print(f"[THREADS] This will keep {max_threads} training tasks running simultaneously")
    print(f"[THREADS] As each task completes, a new one will start immediately")
    
    # Warn if more threads than physical cores
    if max_threads > physical_cores:
        print(f"[WARNING] You have {max_threads} threads but only {physical_cores} physical cores!")
        print(f"[WARNING] Some threads will share physical cores, reducing parallelization efficiency")
        print(f"[SUGGESTION] Consider reducing max_threads to {physical_cores} for optimal performance")
    
    # Reset core assignment counter for this sweep
    global core_assignment_counter
    with core_assignment_lock:
        core_assignment_counter = 0
    
    # Use ThreadPoolExecutor for better thread management
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Give GUI reference to executor for cleanup
        plotter.set_executor(executor)
        # Submit all tasks to the thread pool
        future_to_config = {}
        submitted_configs = list(param_combinations)  # Keep track of submitted configs
        
        print(f"[POOL] Submitting {len(param_combinations)} training tasks to thread pool...")
        
        # Submit initial configurations with priority support
        submitted_configs = list(param_combinations)
        
        def submit_configuration(layer_config):
            """Submit a configuration and track it"""
            # Ensure GUI is properly initialized
            if not plotter.gui:
                print("[ERROR] GUI not initialized when submitting configuration!")
                return None, None
            
            future = executor.submit(
                run_model_threaded, 
                layer_config, 
                database, sweep_config, results_queue, data_queue, plotter.gui, onnx_dir
            )
            future_to_config[future] = layer_config
            
            # Track future for this model
            total_layers = layer_config[-1]
            active_neurons = layer_config[:-1][:total_layers]
            if total_layers == 1:
                model_id = f"{active_neurons[0]}"
            else:
                model_id = "x".join(map(str, active_neurons))
            plotter.gui.model_futures[model_id] = future
            
            return future, model_id
        
        # Ensure GUI is ready before submitting
        while not plotter.gui:
            time.sleep(0.5)
        
        # Submit initial configurations
        for i, layer_config in enumerate(param_combinations):
            result = submit_configuration(layer_config)
            if result[0] is None:  # Failed to submit
                continue
            future, model_id = result
            
            # Log when first batch starts
            if i < max_threads:
                print(f"[START] Model {model_id} starting in thread pool")
        
        # Process completed tasks as they finish
        completed_count = 0
        total_count = len(param_combinations)
        
        print(f"[POOL] All {total_count} training tasks submitted to thread pool")
        print(f"[POOL] First {min(max_threads, total_count)} tasks are now running")
        
        # Dynamic loop that can handle new configurations - keeps running until GUI closes
        loop_count = 0
        sweep_completed = False
        
        while True:
            # Check for global shutdown signal (could be set by GUI)
            global shutdown_requested
            if shutdown_requested or (hasattr(__main__, 'shutdown_requested') and __main__.shutdown_requested):
                print("[LOOP] Shutdown requested, terminating sweep loop")
                # Ensure both flags are set
                with shutdown_lock:
                    shutdown_requested = True
                    if hasattr(__main__, 'shutdown_requested'):
                        __main__.shutdown_requested = True
                break
                
            # Check if GUI is still alive
            if not plotter.gui_thread.is_alive():
                print("[LOOP] GUI closed, setting shutdown flag and terminating")
                with shutdown_lock:
                    shutdown_requested = True
                    if hasattr(__main__, 'shutdown_requested'):
                        __main__.shutdown_requested = True
                break
            
            # Check if we have active work or pending configurations
            has_active_work = bool(future_to_config)
            has_pending_configs = bool(plotter.gui.pending_configurations)
            
            if not has_active_work and not has_pending_configs:
                if not sweep_completed:
                    print(f"[SWEEP] Initial sweep completed - {completed_count} models finished")
                    print(f"[SWEEP] Loop continues running for new configurations...")
                    sweep_completed = True
                
                # No work but keep loop alive for new configurations
                time.sleep(1)  # Check every second when idle
                continue
            loop_count += 1
            # Submit any new configurations from GUI
            
            # Submit new configurations from GUI
            if plotter.gui.pending_configurations:
                # Check available slots
                active_futures = [f for f in future_to_config if not f.done()]
                available_slots = max_threads - len(active_futures)
                
                if available_slots > 0:
                    new_configs = plotter.gui.pending_configurations.copy()
                    plotter.gui.pending_configurations.clear()
                    print(f"[POOL] Processing {len(new_configs)} new configurations, {available_slots} slots available")
                    
                    configs_submitted = 0
                    for config in new_configs:
                        if configs_submitted >= available_slots:
                            # Put back remaining configs
                            plotter.gui.pending_configurations.extend(new_configs[configs_submitted:])
                            print(f"[POOL] No more slots, {len(new_configs) - configs_submitted} configs deferred")
                            break
                        
                        # Skip deleted configurations
                        total_layers = config[-1]
                        active_neurons = config[:-1][:total_layers]
                        if total_layers == 1:
                            check_model_id = f"{active_neurons[0]}"
                        else:
                            check_model_id = "x".join(map(str, active_neurons))
                        
                        if plotter.gui.is_model_deleted(check_model_id):
                            print(f"[SKIP] {check_model_id} was deleted, skipping")
                            continue
                        
                        result = submit_configuration(config)
                        if result[0] is None:  # Failed to submit
                            print(f"[ERROR] Failed to submit {check_model_id}")
                            continue
                        
                        future, model_id = result
                        submitted_configs.append(config)
                        total_count += 1
                        configs_submitted += 1
                        print(f"[NEW CONFIG] Submitted {model_id} to thread pool ({available_slots - configs_submitted} slots remaining)")
                else:
                    # All threads busy - no need to spam console with this info
                    pass
            
            # Process completed futures
            completed_futures = [f for f in future_to_config if f.done()]
            for future in completed_futures:
                config = future_to_config[future]
                total_layers = config[-1]
                active_neurons = config[:-1][:total_layers]
                if total_layers == 1:
                    model_id = f"{active_neurons[0]}"
                else:
                    model_id = "x".join(map(str, active_neurons))
                
                try:
                    result = future.result()
                    completed_count += 1
                    remaining = total_count - completed_count
                    active_threads = len(future_to_config)
                    elapsed = time.time() - start_time
                    
                    # Resource monitoring
                    memory_mb = psutil.virtual_memory().used / 1024 / 1024
                    cpu_percent = psutil.cpu_percent()
                    
                    print(f"[PROGRESS] Completed {completed_count}/{total_count} - Model {model_id} finished")
                    print(f"[THREADS] Active: {active_threads}, Remaining: {remaining}, Elapsed: {elapsed:.1f}s")
                    print(f"[RESOURCES] Memory: {memory_mb:.0f}MB, CPU: {cpu_percent:.1f}%")
                    if remaining > 0 or plotter.gui.pending_configurations:
                        print(f"[POOL] {remaining} tasks remaining, checking for new configurations...")
                    
                except Exception as e:
                    completed_count += 1
                    remaining = total_count - completed_count
                    print(f"[ERROR] Model {model_id} failed with exception: {str(e)}")
                
                # Remove from tracking
                del future_to_config[future]
                
                # Remove from model futures tracking
                if model_id in plotter.gui.model_futures:
                    del plotter.gui.model_futures[model_id]
                
                # Update plot periodically (every 5 completions)
                if completed_count % 5 == 0:
                    plotter.check_and_update_plot()
            
            # Reset sweep_completed flag if we have new work
            if (has_active_work or has_pending_configs) and sweep_completed:
                sweep_completed = False
                print(f"[SWEEP] New work detected - sweep resuming")
            
            # Check for force flag or small delay to prevent busy waiting
            if not completed_futures:
                if plotter.gui.force_check_flag:
                    plotter.gui.force_check_flag = False
                elif plotter.gui.pending_configurations:
                    time.sleep(0.1)  # Check very quickly for new configs
                else:
                    time.sleep(0.5)  # Normal delay when no pending configs
        
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
            print(f"  -> {config}")
    
    # Show best performing models
    successful_results = [(model_id, result) for model_id, success, result in results if success and result]
    if successful_results:
        print(f"\n[RANKINGS] Top performing models (by validation Fidelity):")
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
    
    # Wait for user to close the GUI
    try:
        plotter.wait_for_gui_close()
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed. Closing GUI...")
        plotter.close_plot()
    
    print("[INFO] GUI closed. Program terminating.")

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
    # Make shutdown flag accessible to GUI
    import __main__
    __main__.shutdown_requested = shutdown_requested
    __main__.shutdown_lock = shutdown_lock
    
    print("Multi-Thread Neural Network Architecture Sweep - Real-Time Performance Monitor")
    print("Initializing comprehensive model evaluation...")
    run_sweep_threading()