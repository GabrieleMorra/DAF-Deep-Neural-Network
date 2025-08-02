import tkinter as tk
from tkinter import ttk
import threading
import time
import math
from collections import defaultdict
from queue import Queue, Empty

class RealTimeTableGUI:
    def __init__(self, param_combinations, max_epochs=15000, config_path="NeuralNetworkSweep.json"):
        self.param_combinations = param_combinations
        self.max_epochs = max_epochs
        # Initialize model data with dynamic R2 scores for multiple outputs
        num_outputs = len(self._get_output_variables_preview(config_path))
        default_r2_scores = [0.0] * num_outputs
        default_max_r2_scores = [0.0] * num_outputs
        default_max_r2_epochs = [0] * num_outputs
        self.model_data = defaultdict(lambda: {
            'epoch': 0, 
            'r2_scores': default_r2_scores.copy(), 
            'max_r2_scores': default_max_r2_scores.copy(),
            'max_r2_epochs': default_max_r2_epochs.copy(),
            'optimal_stop_epoch': 0,
            'completed': False, 
            'error': False
        })
        
        # Load dataset information dynamically
        self.dataset_info = self._load_dataset_info(config_path)
        
        # Create architecture labels for each configuration (generic approach)
        self.model_labels = {}
        for layer_config in param_combinations:
            total_layers = layer_config[-1]
            active_neurons = layer_config[:-1][:total_layers]
            
            # Generate model_id consistent with DNN_architecture_sweep.py
            if total_layers == 1:
                model_id = f"{active_neurons[0]}"
                label_text = f"{active_neurons[0]}"
            else:
                model_id = "x".join(map(str, active_neurons))
                label_text = "x".join(map(str, active_neurons))
                
            self.model_labels[model_id] = label_text
        
        self.setup_gui()
    
    def _get_output_variables_preview(self, config_path):
        """Get output variables count for initialization"""
        import json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config["NeuralNetworkModel"]["outputEntryIndices"]
        except:
            return [0]  # Default to 1 output
    
    def _load_dataset_info(self, config_path):
        """Load dataset information from config and CSV files"""
        import json
        import csv
        import os
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get dataset file info
            csv_filename = config["NeuralNetworkModel"]["InputFileName"]
            delimiter = config["NeuralNetworkModel"]["Delimiter"]
            output_indices = config["NeuralNetworkModel"]["outputEntryIndices"]
            
            # Read CSV header
            if os.path.exists(csv_filename):
                with open(csv_filename, 'r') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    headers = next(reader)
                    
                # Extract dataset name from filename
                dataset_name = os.path.splitext(os.path.basename(csv_filename))[0].replace('_', ' ')
                
                # Get output variable names
                output_vars = []
                for idx in output_indices:
                    if idx < len(headers):
                        # Clean the header (remove type info like " REAL")
                        var_name = headers[idx].split()[0] if ' ' in headers[idx] else headers[idx]
                        output_vars.append(var_name)
                
                return {
                    'dataset_name': dataset_name,
                    'filename': csv_filename,
                    'output_variables': output_vars,
                    'headers': headers
                }
            else:
                # Fallback if file doesn't exist
                return {
                    'dataset_name': 'Unknown Dataset',
                    'filename': csv_filename,
                    'output_variables': ['Unknown Variable'],
                    'headers': []
                }
                
        except Exception as e:
            print(f"Warning: Could not load dataset info: {e}")
            return {
                'dataset_name': 'Dataset',
                'filename': 'data.csv',
                'output_variables': ['Target Variable'],
                'headers': []
            }
        
    def setup_gui(self):
        # Main window with modern styling
        self.root = tk.Tk()
        self.root.title("Deep Neural Network Architecture Sweep - Real-Time Training Monitor")
        self.root.geometry("1000x750")
        self.root.configure(bg='#f8f9fa')
        
        # Professional academic color palette
        self.colors = {
            'primary': '#64b5f6',      # Light blue
            'secondary': '#90caf9',    # Lighter blue
            'accent': '#42a5f5',       # Medium blue accent
            'success': '#2e7d32',      # Professional green
            'warning': '#f57c00',      # Academic orange
            'danger': '#d32f2f',       # Professional red
            'dark': '#263238',         # Dark blue-gray
            'light': '#eceff1',        # Light gray
            'background': '#fafafa',   # Clean white
            'header_bg': '#64b5f6',
            'table_bg': '#ffffff'
        }
        
        # Title with gradient-like effect
        title_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        # Dynamic title based on dataset
        title_text = f"Deep Neural Network Architecture Sweep - {self.dataset_info['dataset_name']}"
        title_label = tk.Label(title_frame, text=title_text, 
                              font=('Segoe UI', 18, 'bold'), fg='white', bg=self.colors['primary'])
        title_label.pack(pady=(15, 5))
        
        # Academic subtitle with dataset info
        subtitle_text = f"Real-Time Training Progress Monitor | Dataset: {self.dataset_info['filename']}"
        subtitle_label = tk.Label(title_frame, text=subtitle_text, 
                                 font=('Segoe UI', 11), fg=self.colors['light'], bg=self.colors['primary'])
        subtitle_label.pack()
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Table container with shadow effect
        table_container = tk.Frame(main_container, bg='white', relief='flat', bd=0)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        # Add some visual depth
        shadow_frame = tk.Frame(main_container, bg='#e0e0e0', height=2)
        shadow_frame.pack(fill=tk.X, pady=(0, 3))
        
        # Table header info
        header_info = tk.Frame(table_container, bg='white', height=40)
        header_info.pack(fill=tk.X, padx=15, pady=(15, 10))
        header_info.pack_propagate(False)
        
        info_label = tk.Label(header_info, text="Architecture Performance Analysis", 
                             font=('Segoe UI', 12, 'bold'), fg=self.colors['dark'], bg='white')
        info_label.pack(side=tk.LEFT, pady=10)
        
        # Add professional accent line
        accent_line = tk.Frame(header_info, bg=self.colors['primary'], height=2)
        accent_line.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(20, 0), pady=16)
        
        # Configure modern Treeview style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure professional Treeview colors and fonts
        style.configure("Modern.Treeview",
                       background='white',
                       foreground=self.colors['dark'],
                       rowheight=28,
                       fieldbackground='white',
                       font=('Segoe UI', 10),
                       borderwidth=1,
                       relief='solid')
        
        style.configure("Modern.Treeview.Heading",
                       background=self.colors['primary'],
                       foreground='white',
                       font=('Segoe UI', 11, 'bold'),
                       relief='flat',
                       borderwidth=1)
        
        style.map("Modern.Treeview.Heading",
                 background=[('active', self.colors['accent'])],
                 foreground=[('active', 'white')])
        
        # Professional alternating row colors
        style.map("Modern.Treeview",
                 background=[('selected', self.colors['secondary'])],
                 foreground=[('selected', 'white')])
        
        # Create compact columns - ID first, then architecture, current R² values, summary info, and optimal stop
        id_columns = ['ID']
        base_columns = ['Architecture', 'Epoch']
        output_columns = [f'{var}_R2' for var in self.dataset_info['output_variables']]
        
        # Only show average if multiple outputs
        if len(self.dataset_info['output_variables']) > 1:
            summary_columns = ['Avg_Max_R2', 'Best_Epoch', 'Optimal_Stop']
        else:
            summary_columns = ['Max_R2', 'Best_Epoch', 'Optimal_Stop']
            
        status_columns = ['Status']
        columns = tuple(id_columns + base_columns + output_columns + summary_columns + status_columns)
        
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', 
                                height=18, style="Modern.Treeview")
        
        # Define compact professional headings with sorting
        self.sort_orders = {}  # Track sort order for each column
        self.current_sort_column = None  # Track currently active sort column
        self.current_sort_ascending = True  # Track current sort direction
        
        self.tree.heading('#1', text='ID', command=lambda: self.sort_column('ID'))
        self.tree.heading('#2', text='Architecture', command=lambda: self.sort_column('Architecture'))
        self.tree.heading('#3', text='Epoch', command=lambda: self.sort_column('Epoch'))
        
        col_num = 4
        # Current R² for each output variable
        for i, output_var in enumerate(self.dataset_info['output_variables']):
            col_name = f'{output_var}_R2'
            self.tree.heading(f'#{col_num}', text=f'{output_var} R²', command=lambda c=col_name: self.sort_column(c))
            col_num += 1
        
        # Summary columns
        if len(self.dataset_info['output_variables']) > 1:
            self.tree.heading(f'#{col_num}', text='Avg Max R²', command=lambda: self.sort_column('Avg_Max_R2'))
        else:
            self.tree.heading(f'#{col_num}', text='Max R²', command=lambda: self.sort_column('Max_R2'))
        col_num += 1
        self.tree.heading(f'#{col_num}', text='Best Epoch', command=lambda: self.sort_column('Best_Epoch'))
        col_num += 1
        self.tree.heading(f'#{col_num}', text='Stop Epoch', command=lambda: self.sort_column('Optimal_Stop'))
        col_num += 1
        
        # Status column
        self.tree.heading(f'#{col_num}', text='Status', command=lambda: self.sort_column('Status'))
        
        # Configure compact column widths
        self.tree.column('#1', width=50, anchor='center')   # ID
        self.tree.column('#2', width=100, anchor='center')  # Architecture
        self.tree.column('#3', width=80, anchor='center')   # Epoch
        
        col_num = 4
        # Configure current R² columns
        for i in range(len(self.dataset_info['output_variables'])):
            self.tree.column(f'#{col_num}', width=80, anchor='center')
            col_num += 1
        
        # Configure summary columns
        self.tree.column(f'#{col_num}', width=90, anchor='center')  # Avg Max R²
        col_num += 1
        self.tree.column(f'#{col_num}', width=80, anchor='center')  # Best Epoch
        col_num += 1
        self.tree.column(f'#{col_num}', width=80, anchor='center')  # Stop Epoch
        col_num += 1
        
        # Configure status column
        self.tree.column(f'#{col_num}', width=90, anchor='center')
        
        # Configure professional status colors
        self.tree.tag_configure('waiting', foreground=self.colors['warning'], font=('Segoe UI', 10))
        self.tree.tag_configure('training', foreground='#1976d2', font=('Segoe UI', 10, 'bold'))  # Will be overridden by blink
        self.tree.tag_configure('completed', foreground='#4caf50', font=('Segoe UI', 10, 'bold'))  # Green for completed
        self.tree.tag_configure('error', foreground=self.colors['danger'], font=('Segoe UI', 10, 'bold'))
        
        # For blinking effect on training status
        self.training_blink_state = False
        self.setup_training_blink()
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        # Pack table and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 15))
        
        # Initialize table rows with dynamic values
        self.table_items = {}
        for idx, layer_config in enumerate(self.param_combinations):
            total_layers = layer_config[-1]
            active_neurons = layer_config[:-1][:total_layers]
            
            if total_layers == 1:
                model_id = f"{active_neurons[0]}"
            else:
                model_id = "x".join(map(str, active_neurons))
            arch_label = self.model_labels[model_id]
            
            # Store ID in model data for efficiency
            self.model_data[model_id]['id'] = idx
            
            # Create initial values: ID, Architecture, Epoch, Current R² scores, Summary columns, Status
            initial_values = [str(idx), arch_label, '0']
            initial_values.extend(['0.00'] * len(self.dataset_info['output_variables']))  # Current R² scores
            initial_values.extend(['0.00', '0', '0'])  # Max R² (or Avg Max R²), Best Epoch, Stop Epoch
            initial_values.append('Waiting')  # Status
            
            # Insert row
            item = self.tree.insert('', 'end', values=tuple(initial_values))
            self.table_items[model_id] = item
        
        # Professional status bar
        status_frame = tk.Frame(self.root, bg=self.colors['dark'], height=45)
        status_frame.pack(fill=tk.X, padx=0, pady=0)
        status_frame.pack_propagate(False)
        
        # Add professional accent line
        accent_frame = tk.Frame(status_frame, bg=self.colors['primary'], height=2)
        accent_frame.pack(fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="Initializing training environment...", 
                                    font=('Segoe UI', 11), fg='white', bg=self.colors['dark'])
        self.status_label.pack(expand=True, pady=10)
        
        # Setup window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_training_blink(self):
        """Setup professional blinking effect for training status"""
        def blink():
            self.training_blink_state = not self.training_blink_state
            if self.training_blink_state:
                # Light blue for training
                self.tree.tag_configure('training', foreground='#90caf9', 
                                      font=('Segoe UI', 10, 'bold'))  # Light blue
            else:
                # Medium blue for training
                self.tree.tag_configure('training', foreground='#1976d2', 
                                      font=('Segoe UI', 10, 'bold'))  # Medium blue
            
            # Schedule next blink
            self.root.after(800, blink)  # Professional blink interval
        
        # Start blinking
        self.root.after(800, blink)
        
    def add_data_point(self, model_id, epoch, r2_scores):
        """Add a data point and update the table"""
        self.model_data[model_id]['epoch'] = epoch
        
        # Handle both single value and list of values
        if isinstance(r2_scores, (list, tuple)):
            current_r2 = list(r2_scores)
            self.model_data[model_id]['r2_scores'] = current_r2
        else:
            # Single value - put it in the first position, pad others with 0
            current_r2 = [r2_scores] + [0.0] * (len(self.model_data[model_id]['r2_scores']) - 1)
            self.model_data[model_id]['r2_scores'] = current_r2
        
        # Update maximum R² values and epochs
        for i, r2_val in enumerate(current_r2):
            if r2_val > self.model_data[model_id]['max_r2_scores'][i]:
                self.model_data[model_id]['max_r2_scores'][i] = r2_val
                self.model_data[model_id]['max_r2_epochs'][i] = epoch
        
        # Calculate summary statistics
        max_epochs = self.model_data[model_id]['max_r2_epochs']
        max_r2_scores = self.model_data[model_id]['max_r2_scores']
        
        if any(ep > 0 for ep in max_epochs):
            # Best epoch is when the best overall performance was achieved
            best_epoch = max(max_epochs)
            # Optimal stop epoch with patience (20% more)
            optimal_stop = int(best_epoch * 1.2)
            # Average of maximum R² scores achieved
            avg_max_r2 = sum(max_r2_scores) / len(max_r2_scores) if max_r2_scores else 0
            
            self.model_data[model_id]['optimal_stop_epoch'] = optimal_stop
            self.model_data[model_id]['best_epoch'] = best_epoch
            self.model_data[model_id]['avg_max_r2'] = avg_max_r2
        
        # Update table in main thread
        self.root.after(0, self.update_table_row, model_id)
        
    def mark_completed(self, model_id):
        """Mark a model as completed"""
        self.model_data[model_id]['completed'] = True
        self.root.after(0, self.update_table_row, model_id)
    
    def mark_error(self, model_id):
        """Mark a model as failed"""
        self.model_data[model_id]['error'] = True
        self.root.after(0, self.update_table_row, model_id)
    
    def update_table_row(self, model_id):
        """Update a specific row in the table"""
        if model_id not in self.table_items:
            return
            
        data = self.model_data[model_id]
        arch_label = self.model_labels[model_id]
        
        # Determine status and tag
        if data['error']:
            status = 'Error'
            tag = 'error'
        elif data['completed']:
            status = 'Completed'
            tag = 'completed'
        elif data['epoch'] > 0:
            status = 'Training'
            tag = 'training'
        else:
            status = 'Waiting'
            tag = 'waiting'
        
        # Build values: ID, Architecture, Epoch, Current R² scores, Summary stats, Status
        model_index = data.get('id', '?')
        values = [str(model_index), arch_label, f"{data['epoch']}"]
        
        # Add current R² scores for each output variable
        for i, r2_score in enumerate(data['r2_scores']):
            values.append(f"{r2_score:.2f}")
        
        # Add summary statistics
        best_epoch = data.get('best_epoch', 0)
        optimal_stop = data.get('optimal_stop_epoch', 0)
        
        # Show either average or single max R² depending on number of outputs
        if len(self.dataset_info['output_variables']) > 1:
            summary_r2 = data.get('avg_max_r2', 0)
        else:
            summary_r2 = data.get('max_r2_scores', [0])[0] if data.get('max_r2_scores') else 0
        
        values.extend([f"{summary_r2:.2f}", f"{best_epoch}", f"{optimal_stop}"])
        values.append(status)
        
        # Update the row with appropriate color tag
        self.tree.item(self.table_items[model_id], 
                      values=tuple(values),
                      tags=(tag,))
        
        # If there's an active sort, re-apply it to keep the table sorted
        if self.current_sort_column is not None:
            self._perform_sort(self.current_sort_column, self.current_sort_ascending)
        
        # Update status
        self.update_status()
    
    def update_status(self):
        """Update the status bar"""
        completed = len([m for m in self.model_data.values() if m['completed']])
        training = len([m for m in self.model_data.values() if m['epoch'] > 0 and not m['completed'] and not m['error']])
        waiting = len([m for m in self.model_data.values() if m['epoch'] == 0 and not m['error']])
        errors = len([m for m in self.model_data.values() if m['error']])
        total = len(self.param_combinations)
        
        status_text = f"Total: {total} | Training: {training} | Completed: {completed} | Waiting: {waiting} | Errors: {errors}"
        self.status_label.config(text=status_text)
        
    def collect_training_data(self, data_queue):
        """Collect data from training threads"""
        while True:
            try:
                data = data_queue.get(timeout=1)
                if data is None:
                    break
                
                if len(data) == 3:
                    # Regular training data
                    model_id, epoch, r2_scores = data
                    self.add_data_point(model_id, epoch, r2_scores)
                elif len(data) == 2 and data[1] == 'COMPLETED':
                    # Completion marker
                    model_id = data[0]
                    self.mark_completed(model_id)
                elif len(data) == 2 and data[1] == 'ERROR':
                    # Error marker
                    model_id = data[0]
                    self.mark_error(model_id)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in data collection: {e}")
                continue
                
    def show(self):
        """Show the GUI"""
        self.root.mainloop()
        
    def close(self):
        """Close the GUI safely from any thread"""
        if self.root:
            try:
                self.root.after(0, self._safe_close)
            except:
                # If we can't schedule, try direct close
                try:
                    self.root.quit()
                except:
                    pass
    
    def on_closing(self):
        """Handle window close event - terminate all processes"""
        import os
        import sys
        print("\nGUI closing - terminating all training processes...")
        
        # Force terminate the entire process tree
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
        
        # Force exit the entire program
        os._exit(0)
    
    def _safe_close(self):
        """Internal method to safely close GUI"""
        self.on_closing()
    
    def sort_column(self, col_name):
        """Sort table by column with alternating ascending/descending order"""
        # Toggle sort order for this column
        if col_name not in self.sort_orders:
            self.sort_orders[col_name] = True  # True = ascending, False = descending
        else:
            self.sort_orders[col_name] = not self.sort_orders[col_name]
        
        ascending = self.sort_orders[col_name]
        
        # Update current sort tracking
        self.current_sort_column = col_name
        self.current_sort_ascending = ascending
        
        # Perform the actual sorting
        self._perform_sort(col_name, ascending)
    
    def _perform_sort(self, col_name, ascending):
        """Internal method to perform the actual sorting"""
        # Get all items with their values
        items = []
        for item in self.tree.get_children():
            values = self.tree.item(item, 'values')
            items.append((item, values))
        
        # Find column index
        columns = ['ID', 'Architecture', 'Epoch'] + [f'{var}_R2' for var in self.dataset_info['output_variables']]
        if len(self.dataset_info['output_variables']) > 1:
            columns.extend(['Avg_Max_R2', 'Best_Epoch', 'Optimal_Stop'])
        else:
            columns.extend(['Max_R2', 'Best_Epoch', 'Optimal_Stop'])
        columns.append('Status')
        
        try:
            col_index = columns.index(col_name)
        except ValueError:
            return  # Column not found
        
        # Sort based on column type
        def sort_key(item):
            value = item[1][col_index]
            
            # Handle different data types
            if col_name == 'ID':
                return int(value) if value.isdigit() else 0
            elif col_name in ['Epoch', 'Best_Epoch', 'Optimal_Stop']:
                return int(value) if value.isdigit() else 0
            elif '_R2' in col_name or 'Max_R2' in col_name:
                try:
                    return float(value)
                except ValueError:
                    return 0.0
            elif col_name == 'Architecture':
                # Sort architectures logically: single numbers first, then combinations
                if 'x' in value:
                    # For combinations like "5x10", sort by first number then second
                    parts = value.split('x')
                    return (1000 + int(parts[0]), int(parts[1]))  # Offset to put after singles
                else:
                    # For single numbers like "5", sort by number
                    return (int(value), 0)
            else:
                return str(value).lower()
        
        # Sort items
        items.sort(key=sort_key, reverse=not ascending)
        
        # Clear tree and re-insert sorted items
        for item, _ in items:
            self.tree.move(item, '', 'end')
        
        # Update column heading to show sort direction
        direction = " ↑" if ascending else " ↓"
        current_text = self.tree.heading(f'#{col_index + 1}')['text']
        # Remove any existing direction indicators
        clean_text = current_text.replace(" ↑", "").replace(" ↓", "")
        self.tree.heading(f'#{col_index + 1}', text=clean_text + direction)
        
        # Clear direction indicators from other columns
        for i, col in enumerate(columns):
            if col != col_name:
                heading_text = self.tree.heading(f'#{i + 1}')['text']
                clean_text = heading_text.replace(" ↑", "").replace(" ↓", "")
                self.tree.heading(f'#{i + 1}', text=clean_text)

# Test the GUI
if __name__ == "__main__":
    # Test with sample configurations
    test_configs = [(5, 10, 2), (10, 20, 3), (15, 30, 2)]
    
    gui = RealTimeTableGUI(test_configs)
    
    # Add some test data
    def add_test_data():
        for i in range(20):
            for j, (neurons_input, neurons_hidden, num_layers) in enumerate(test_configs):
                if num_layers == 1:
                    model_id = f"{neurons_input}"
                else:
                    model_id = f"{neurons_input}x{neurons_hidden}"
                epoch = i * 250
                # Test single output (current case)
                r2_score = 30 + 40 * math.sin(i * 0.3 + j) + 20
                gui.add_data_point(model_id, epoch, r2_score)
                time.sleep(0.2)
        
        # Mark some as completed and one as error
        config0 = test_configs[0]
        config1 = test_configs[1] 
        config2 = test_configs[2]
        
        model_id0 = f"{config0[0]}" if config0[2] == 1 else f"{config0[0]}x{config0[1]}"
        model_id1 = f"{config1[0]}" if config1[2] == 1 else f"{config1[0]}x{config1[1]}" 
        model_id2 = f"{config2[0]}" if config2[2] == 1 else f"{config2[0]}x{config2[1]}"
        
        gui.mark_completed(model_id0)
        time.sleep(2)
        gui.mark_error(model_id1)
        time.sleep(1)
        gui.mark_completed(model_id2)
    
    # Start test data in separate thread
    test_thread = threading.Thread(target=add_test_data)
    test_thread.daemon = True
    test_thread.start()
    
    gui.show()