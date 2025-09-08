import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
from collections import defaultdict
from queue import Queue, Empty

class RealTimeTableGUI:
    def configure_optimal_column_widths(self):
        """Calculate optimal column widths based on header text and adjust window size"""
        import tkinter.font as tkFont
        
        # Get font for measuring text width
        header_font = tkFont.Font(family='Segoe UI', size=10, weight='bold')
        
        # Define minimum widths for different column types
        min_widths = {
            'ID': 40,
            'Architecture': 80,
            'Epoch': 60,
            'R2': 70,
            'Summary': 80,
            'Status': 70
        }
        
        # Get all column headers
        headers = []
        for col in self.tree['columns']:
            header_text = self.tree.heading(col)['text']
            headers.append(header_text)
        
        # Calculate widths for each column
        column_widths = []
        total_width = 0
        
        for i, header in enumerate(headers):
            col_id = f'#{i+1}'
            
            # Measure header text width
            text_width = header_font.measure(header)
            
            # Add padding (25px extra for borders, sorting arrows, etc.)
            optimal_width = text_width + 25
            
            # Apply minimum width based on column type
            if header == 'ID':
                optimal_width = max(optimal_width, min_widths['ID'])
            elif header == 'Architecture':
                optimal_width = max(optimal_width, min_widths['Architecture'])
            elif header == 'Epoch':
                optimal_width = max(optimal_width, min_widths['Epoch'])
            elif 'R²' in header or 'R2' in header:
                optimal_width = max(optimal_width, min_widths['R2'])
            elif header == 'Status':
                optimal_width = max(optimal_width, min_widths['Status'])
            else:
                optimal_width = max(optimal_width, min_widths['Summary'])
            
            column_widths.append(optimal_width)
            total_width += optimal_width
            
            # Set column width
            self.tree.column(col_id, width=optimal_width, anchor='center')
        
        # Adjust window width to fit all columns
        self.adjust_window_size(total_width)
    
    def adjust_window_size(self, table_width):
        """Adjust window size to fit table while respecting screen limits"""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate required window width (table + margins + scrollbars)
        margin_width = 50  # Left/right margins
        scrollbar_width = 20  # Vertical scrollbar
        side_panel_width = 420  # Right configuration panel
        
        required_width = table_width + margin_width + scrollbar_width + side_panel_width
        
        # Limit to 95% of screen width (consistent with setup_gui)
        max_window_width = int(screen_width * 0.95)
        optimal_width = min(required_width, max_window_width)
        
        # Set minimum width to ensure usability and scrollbar visibility
        min_window_width = 1300  # Increased to give space for vertical scrollbar
        window_width = max(optimal_width, min_window_width)
        
        # Set height as percentage of screen resolution
        desired_height = int(screen_height * 0.85)  # 85% of screen height
        max_window_height = int(screen_height * 0.9)  # Keep 90% for height (taskbar space)
        window_height = min(desired_height, max_window_height)
        
        # Position window higher on screen instead of center
        x = (screen_width - window_width) // 2
        y = max(50, (screen_height - window_height) // 4)  # Position at 1/4 from top, minimum 50px from top
        
        # Update window geometry
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        print(f"[GUI] Window adjusted to {window_width}x{window_height} (table width: {table_width}px)")
        
        # Store if horizontal scrolling is needed
        self.horizontal_scroll_needed = required_width > max_window_width
        if self.horizontal_scroll_needed:
            print(f"[GUI] Table too wide for screen, horizontal scrolling enabled")
            
        # Store if vertical scrolling is needed (more than table height rows)
        self.update_scrollbar_visibility()
    
    def update_scrollbar_visibility(self):
        """Update scrollbar visibility based on current content"""
        table_height = 8  # Height defined in create_table (reduced from 12 to 8)
        
        # Calculate total rows: original + pending
        original_rows = len(self.param_combinations)
        pending_rows = len(getattr(self, 'pending_configurations', []))
        total_rows = original_rows + pending_rows
        
        # Update vertical scrolling flag
        self.vertical_scroll_needed = total_rows > table_height
        if self.vertical_scroll_needed:
            print(f"[GUI] Vertical scrolling enabled ({total_rows} rows > {table_height} visible)")
            
        # Update scrollbar visibility if scrollbars exist
        if hasattr(self, 'v_scrollbar') and hasattr(self, 'h_scrollbar'):
            # Hide all scrollbars first
            self.v_scrollbar.pack_forget()
            self.h_scrollbar.pack_forget()
            
            # Show scrollbars if needed
            if getattr(self, 'horizontal_scroll_needed', False):
                self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=(15, 15), pady=(0, 15))
                
            if self.vertical_scroll_needed:
                self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 15), padx=(0, 15))
    
    def draw_dnn_logo(self, canvas):
        """Draw a clean DNN architecture logo"""
        # Logo colors
        node_color = '#FFFFFF'
        connection_color = '#E8F4FD'
        
        # Input layer (3 nodes)
        input_nodes = [(15, 15), (15, 30), (15, 45)]
        
        # Hidden layer 1 (4 nodes)
        hidden1_nodes = [(35, 10), (35, 22), (35, 38), (35, 50)]
        
        # Hidden layer 2 (3 nodes)
        hidden2_nodes = [(55, 15), (55, 30), (55, 45)]
        
        # Output layer (2 nodes)
        output_nodes = [(70, 22), (70, 38)]
        
        # Draw connections
        # Input to Hidden1
        for input_pos in input_nodes:
            for hidden_pos in hidden1_nodes:
                canvas.create_line(input_pos[0], input_pos[1], 
                                 hidden_pos[0], hidden_pos[1],
                                 fill=connection_color, width=1)
        
        # Hidden1 to Hidden2
        for h1_pos in hidden1_nodes:
            for h2_pos in hidden2_nodes:
                canvas.create_line(h1_pos[0], h1_pos[1], 
                                 h2_pos[0], h2_pos[1],
                                 fill=connection_color, width=1)
        
        # Hidden2 to Output
        for h2_pos in hidden2_nodes:
            for out_pos in output_nodes:
                canvas.create_line(h2_pos[0], h2_pos[1], 
                                 out_pos[0], out_pos[1],
                                 fill=connection_color, width=1)
        
        # Draw nodes
        node_radius = 3
        
        # Input nodes
        for pos in input_nodes:
            canvas.create_oval(pos[0]-node_radius, pos[1]-node_radius,
                             pos[0]+node_radius, pos[1]+node_radius,
                             fill=node_color, outline='#1976D2', width=1)
        
        # Hidden layer 1 nodes
        for pos in hidden1_nodes:
            canvas.create_oval(pos[0]-node_radius, pos[1]-node_radius,
                             pos[0]+node_radius, pos[1]+node_radius,
                             fill=node_color, outline='#1976D2', width=1)
        
        # Hidden layer 2 nodes
        for pos in hidden2_nodes:
            canvas.create_oval(pos[0]-node_radius, pos[1]-node_radius,
                             pos[0]+node_radius, pos[1]+node_radius,
                             fill=node_color, outline='#1976D2', width=1)
        
        # Output nodes (slightly larger)
        output_radius = 4
        for pos in output_nodes:
            canvas.create_oval(pos[0]-output_radius, pos[1]-output_radius,
                             pos[0]+output_radius, pos[1]+output_radius,
                             fill=node_color, outline='#FF6F00', width=2)

    def __init__(self, param_combinations, max_epochs=15000, config_path="NeuralNetworkSweep.json", data_queue=None, pause_state=None, new_config_queue=None, database_info=None):
        self.param_combinations = param_combinations
        self.max_epochs = max_epochs
        self.config_path = config_path
        self.data_queue = data_queue or Queue()
        self.external_pause_state = pause_state  # Reference to external pause state
        self.new_config_queue = new_config_queue  # Queue for sending new configs to training thread
        
        # Flag to stop data collection thread
        self.data_collection_active = True
        
        # Initialize model data with dynamic R2 scores for multiple outputs
        num_outputs = len(self._get_output_variables_preview(config_path))
        default_r2_scores = [0.0] * num_outputs
        self.model_data = defaultdict(lambda: {
            'current_epoch': 0, 
            'best_epoch': 0,
            'best_r2_scores': default_r2_scores.copy(),
            'best_avg_r2': 0.0,
            'best_training_fidelity': 0.0,
            'best_validation_fidelity': 0.0,
            'best_loss': None,  # None = not started, will display as N/A
            'completed': False, 
            'error': False
        })
        
        # Load dataset information and full configuration
        self.dataset_info = self._load_dataset_info(config_path)
        self.config = self._load_full_config(config_path)
        
        # Add database sample counts if provided
        if database_info:
            X_train = database_info.get('X_train')
            X_valid = database_info.get('X_valid')
            
            # Calculate number of input variables
            num_inputs = 0
            if X_train is not None and len(X_train) > 0:
                num_inputs = len(X_train[0]) if hasattr(X_train[0], '__len__') else 0
            
            self.dataset_info.update({
                'train_samples': len(X_train) if X_train is not None else 0,
                'val_samples': len(X_valid) if X_valid is not None else 0,
                'input_variables': list(range(num_inputs))
            })
        
        # Load ShowTestEvery for dashboard update frequency
        self.show_test_every = self.dataset_info.get('show_test_every', 500)
        
        # Control state for sweep management
        self.is_paused = False
        self.executor_ref = None  # Reference to ThreadPoolExecutor
        self.pending_configurations = []  # New configurations to add
        
        # Selection tracking for deselect functionality
        self.last_selected_item = None
        
        # Individual model control
        self.paused_models = set()  # Models individually paused
        self.deleted_models = set()  # Models marked for deletion
        self.model_futures = {}  # Track futures for each model
        self.force_check_flag = False  # Flag to force thread pool check
        self.next_config_id = len(param_combinations)  # Track next available ID
        
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
    
    def _load_full_config(self, config_path):
        """Load complete configuration for access to global settings"""
        import json
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {}  # Return empty dict if config cannot be loaded
    
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
            show_test_every = config["NeuralNetworkModel"].get("ShowTestEvery", 500)
            
            # Try to find CSV file in multiple locations
            csv_paths = [
                csv_filename,  # Current directory
                os.path.join("data", csv_filename),  # data folder
                os.path.join("data", "input", csv_filename),  # data/input folder
            ]
            
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            # Read CSV header
            if csv_path:
                with open(csv_path, 'r') as f:
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
                    'headers': headers,
                    'show_test_every': show_test_every
                }
            else:
                # Fallback if file doesn't exist - try to get info from config only
                print(f"[WARNING] CSV file not found: {csv_filename} (searched in: {csv_paths})")
                
                # Try to use reasonable defaults based on Boston Housing dataset
                if "Boston" in csv_filename or "Housing" in csv_filename:
                    output_vars = ["MEDV"]  # Median value for Boston Housing
                else:
                    output_vars = [f"Output_{i}" for i in output_indices]
                
                return {
                    'dataset_name': os.path.splitext(os.path.basename(csv_filename))[0].replace('_', ' '),
                    'filename': csv_filename,
                    'output_variables': output_vars,
                    'headers': [],
                    'show_test_every': show_test_every
                }
                
        except Exception as e:
            print(f"Warning: Could not load dataset info: {e}")
            return {
                'dataset_name': 'Dataset',
                'filename': 'data.csv',
                'output_variables': ['Target Variable'],
                'headers': [],
                'show_test_every': 500
            }
    
    def _get_config_info(self):
        """Get configuration parameters for display"""
        try:
            import json
            with open(getattr(self, 'config_path', 'NeuralNetworkSweep.json'), 'r') as f:
                config = json.load(f)
            
            # Extract key parameters
            epochs = config["NeuralNetworkModel"]["epochs"]
            learning_rate = config["NeuralNetworkModel"]["learning_rate"]
            max_threads = config["SweepConfiguration"]["max_threads"]
            loss_function = config["NeuralNetworkModel"]["loss"]
            
            # Format learning rate nicely
            if learning_rate >= 0.01:
                lr_str = f"{learning_rate:.3f}"
            elif learning_rate >= 0.001:
                lr_str = f"{learning_rate:.4f}"
            else:
                lr_str = f"{learning_rate:.0e}"
            
            return {
                'epochs': str(epochs),
                'learning_rate': lr_str,
                'max_threads': str(max_threads),
                'loss_function': loss_function
            }
        except Exception as e:
            return {
                'epochs': 'N/A',
                'learning_rate': 'N/A', 
                'max_threads': 'N/A',
                'loss_function': 'N/A'
            }
        
    def setup_gui(self):
        # Main window with modern styling
        self.root = tk.Tk()
        self.root.title("Multi-Thread Neural Network Architecture Sweep - Real-Time Training Monitor")
        
        # Get screen dimensions and limit window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set maximum window size to 95% of screen dimensions
        max_width = int(screen_width * 0.95)
        max_height = int(screen_height * 0.9)
        
        # Configure window
        self.root.configure(bg='#f8f9fa')
        self.root.maxsize(max_width, max_height)
        
        # Set window to a reasonable size based on screen resolution
        initial_height = int(screen_height * 0.85)  # 85% of screen height
        self.root.geometry(f"1400x{initial_height}")  # Responsive height
        
        # Store dimensions for later use
        self.max_window_width = max_width
        self.max_window_height = max_height
        
        # Professional academic color palette
        self.colors = {
            'primary': '#64b5f6',      # Light blue
            'secondary': '#90caf9',    # Lighter blue
            'accent': '#42a5f5',       # Medium blue accent
            'success': '#2e7d32',      # Professional green
            'warning': '#ff9800',      # Professional orange
            'danger': '#d32f2f',       # Professional red
            'dark': '#263238',         # Dark blue-gray
            'light': '#eceff1',        # Light gray
            'background': '#fafafa',   # Clean white
            'header_bg': '#64b5f6',
            'table_bg': '#ffffff'
        }
        
        # Title with gradient-like effect and logo
        title_frame = tk.Frame(self.root, bg=self.colors['primary'], height=100)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        # Create DNN architecture logo
        logo_canvas = tk.Canvas(title_frame, width=80, height=60, bg=self.colors['primary'], highlightthickness=0)
        logo_canvas.pack(side=tk.LEFT, padx=(20, 10), pady=20)
        
        # Draw simple DNN architecture
        self.draw_dnn_logo(logo_canvas)
        
        # Title and subtitle container
        text_container = tk.Frame(title_frame, bg=self.colors['primary'])
        text_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10)
        
        # Static title
        title_text = "Multi-Thread Neural Network Architecture Sweep"
        title_label = tk.Label(text_container, text=title_text, 
                              font=('Segoe UI', 18, 'bold'), fg='white', bg=self.colors['primary'])
        title_label.pack(anchor='w', pady=(5, 2))
        
        # Academic subtitle with dataset info
        subtitle_text = f"Real-Time Training Progress Monitor | Dataset: {self.dataset_info['filename']}"
        subtitle_label = tk.Label(text_container, text=subtitle_text, 
                                 font=('Segoe UI', 11), fg=self.colors['light'], bg=self.colors['primary'])
        subtitle_label.pack(anchor='w')
        
        # Store reference for dashboard positioning
        self.title_frame = title_frame
        
        # Create dashboard panel immediately after title
        self.create_dashboard_panel()
        
        # Main container with padding - reduced expansion to leave more space for controls
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
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
        
        # Create compact columns - ID first, then architecture, epoch, best epoch, then R² values, summary info
        id_columns = ['ID']
        base_columns = ['Architecture', 'Epoch', 'Best_Epoch']
        output_columns = [f'{var}_R2' for var in self.dataset_info['output_variables']]
        
        # Only show average R2 (remove max R2)
        if len(self.dataset_info['output_variables']) > 1:
            summary_columns = ['Avg_R2', 'Training_R2', 'Validation_R2', 'Loss']
        else:
            summary_columns = ['Training_R2', 'Validation_R2', 'Loss']
            
        status_columns = ['Status']
        columns = tuple(id_columns + base_columns + output_columns + summary_columns + status_columns)
        
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', 
                                height=8, style="Modern.Treeview", selectmode='extended')
        
        # Define compact professional headings with sorting
        self.sort_orders = {}  # Track sort order for each column
        self.current_sort_column = None  # Track currently active sort column
        self.current_sort_ascending = True  # Track current sort direction
        
        self.tree.heading('#1', text='ID', command=lambda: self.sort_column('ID'))
        self.tree.heading('#2', text='Architecture', command=lambda: self.sort_column('Architecture'))
        self.tree.heading('#3', text='Epoch', command=lambda: self.sort_column('Epoch'))
        self.tree.heading('#4', text='Best Epoch', command=lambda: self.sort_column('Best_Epoch'))
        
        col_num = 5
        # R² for each output variable (showing best epoch values)
        for i, output_var in enumerate(self.dataset_info['output_variables']):
            col_name = f'{output_var}_R2'
            self.tree.heading(f'#{col_num}', text=f'{output_var} R²', command=lambda c=col_name: self.sort_column(c))
            col_num += 1
        
        # Summary columns - only show average R2 if multiple outputs
        if len(self.dataset_info['output_variables']) > 1:
            self.tree.heading(f'#{col_num}', text='Avg R²', command=lambda: self.sort_column('Avg_R2'))
            col_num += 1
            
        self.tree.heading(f'#{col_num}', text='Training Fidelity', command=lambda: self.sort_column('Training_R2'))
        col_num += 1
        self.tree.heading(f'#{col_num}', text='Validation Fidelity', command=lambda: self.sort_column('Validation_R2'))
        col_num += 1
        self.tree.heading(f'#{col_num}', text='Loss', command=lambda: self.sort_column('Loss'))
        col_num += 1
        
        # Status column
        self.tree.heading(f'#{col_num}', text='Status', command=lambda: self.sort_column('Status'))
        
        # Calculate optimal column widths based on headers
        self.configure_optimal_column_widths()
        
        # Configure professional status colors
        self.tree.tag_configure('waiting', foreground=self.colors['warning'], font=('Segoe UI', 10))
        self.tree.tag_configure('training', foreground='#1976d2', font=('Segoe UI', 10, 'bold'))  # Will be overridden by blink
        self.tree.tag_configure('completed', foreground='#4caf50', font=('Segoe UI', 10, 'bold'))  # Green for completed
        self.tree.tag_configure('error', foreground=self.colors['danger'], font=('Segoe UI', 10, 'bold'))
        self.tree.tag_configure('paused', foreground='#ff9800', font=('Segoe UI', 10, 'italic'))  # Orange italic for paused
        
        # For blinking effect on training status
        self.training_blink_state = False
        self.setup_training_blink()
        
        # Create vertical scrollbar but only show if needed
        self.v_scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.v_scrollbar.set)
        
        # Create horizontal scrollbar but only show if needed
        self.h_scrollbar = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscrollcommand=self.h_scrollbar.set)
        
        # Pack scrollbars first, then table with proper padding
        # Only show scrollbars if determined to be needed earlier
        if getattr(self, 'horizontal_scroll_needed', False):
            self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=(15, 15), pady=(0, 15))
            
        if getattr(self, 'vertical_scroll_needed', False):
            self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 15), padx=(0, 15))
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 0), pady=(0, 15))
        
        # Enhanced event bindings
        self.tree.bind('<ButtonRelease-1>', self.on_tree_click)
        self.tree.bind('<Button-3>', self.on_right_click)  # Right click for context menu
        self.tree.bind('<Motion>', self.on_tree_hover)     # Hover effects
        self.tree.bind('<Leave>', self.on_tree_leave)      # Leave hover
        
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
            
            # Create initial values: ID, Architecture, Epoch, Best Epoch, R² scores, Summary columns, Status
            initial_values = [str(idx), arch_label, '0', '0']  # ID, Architecture, Current Epoch, Best Epoch
            initial_values.extend(['0.00'] * len(self.dataset_info['output_variables']))  # R² scores per variable
            
            # Add Avg R² column only if multiple outputs
            if len(self.dataset_info['output_variables']) > 1:
                initial_values.append('0.00')  # Avg R²
            
            # Add remaining summary columns: Training Fidelity, Validation Fidelity, Loss
            initial_values.extend(['0.00', '0.00', 'N/A'])  # Training, Validation, Loss
            initial_values.append('Waiting')  # Status
            
            # Insert row
            item = self.tree.insert('', 'end', values=tuple(initial_values))
            self.table_items[model_id] = item
        
        # Control buttons frame with more space
        control_frame = tk.Frame(self.root, bg=self.colors['dark'], height=80)
        control_frame.pack(fill=tk.X, padx=0, pady=0)
        control_frame.pack_propagate(False)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg=self.colors['dark'])
        button_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Enhanced professional buttons
        self.pause_button = self.create_control_button(
            button_frame, "■ Pause", self.toggle_pause, self.colors['warning'], "left"
        )
        
        self.add_config_button = self.create_control_button(
            button_frame, "+ Add Models", self.open_add_config_window, self.colors['secondary'], "left", font_size=10
        )
        
        self.export_button = self.create_control_button(
            button_frame, "↓ Export", self.export_table, self.colors['accent'], "left", font_size=10
        )
        
        # Professional status indicator
        self.status_indicator = tk.Label(button_frame, text="● Running", 
                                        font=('Segoe UI', 11, 'bold'),
                                        fg='#4caf50', bg=self.colors['dark'])
        self.status_indicator.pack(side=tk.RIGHT, padx=(20, 0))
    
    def create_control_button(self, parent, text, command, color, side, font_size=10):
        """Create a professional control button"""
        # Adjust padding based on font size
        pad_x = 25 if font_size <= 10 else 30
        pad_y = 8 if font_size <= 10 else 12
        
        button = tk.Button(
            parent, text=text, command=command,
            font=('Segoe UI', font_size, 'bold'),
            bg=color, fg='white',
            relief='flat', padx=pad_x, pady=pad_y,
            cursor='hand2',
            activebackground=self.darken_color(color),
            activeforeground='white',
            bd=0,
            highlightthickness=0
        )
        
        # Add hover effects
        button.bind('<Enter>', lambda e: button.config(bg=self.darken_color(color)))
        button.bind('<Leave>', lambda e: button.config(bg=color))
        
        button.pack(side=getattr(tk, side.upper()), padx=(0, 12))
        return button
    
    def darken_color(self, color):
        """Darken a hex color for hover effects"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        darker_rgb = tuple(max(0, int(c * 0.8)) for c in rgb)
        return f"#{darker_rgb[0]:02x}{darker_rgb[1]:02x}{darker_rgb[2]:02x}"
        
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
        
        # Initialize dashboard with current values
        self.update_status()
    
    def create_dashboard_panel(self):
        """Create professional dashboard with live metrics"""
        dashboard_frame = tk.Frame(self.root, bg='#f5f5f5', height=100)
        dashboard_frame.pack(fill=tk.X, padx=0, pady=0)
        dashboard_frame.pack_propagate(False)
        
        # Add subtle top border
        border_frame = tk.Frame(dashboard_frame, bg='#e0e0e0', height=1)
        border_frame.pack(fill=tk.X)
        
        # Main dashboard container
        dashboard_container = tk.Frame(dashboard_frame, bg='#f5f5f5')
        dashboard_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)
        
        # Metrics cards container
        metrics_container = tk.Frame(dashboard_container, bg='#f5f5f5')
        metrics_container.pack(fill=tk.BOTH, expand=True)
        
        # Create metric cards
        self.metric_cards = {}
        
        # Total Models card
        self.metric_cards['total'] = self.create_metric_card(
            metrics_container, "📊 Total", "0", "#2196F3"
        )
        
        # Training card
        self.metric_cards['training'] = self.create_metric_card(
            metrics_container, "🔄 Training", "0", "#FF9800"
        )
        
        # Completed card
        self.metric_cards['completed'] = self.create_metric_card(
            metrics_container, "✅ Done", "0", "#4CAF50"
        )
        
        # Best Performance card
        self.metric_cards['best_r2'] = self.create_metric_card(
            metrics_container, "🏆 Best R²", "0.00%", "#9C27B0"
        )
        
        # Configuration parameters cards
        config_info = self._get_config_info()
        
        # Epochs card
        self.metric_cards['epochs'] = self.create_metric_card(
            metrics_container, "⚡ Epochs", config_info.get('epochs', 'N/A'), "#FF5722"
        )
        
        # Learning Rate card  
        self.metric_cards['learning_rate'] = self.create_metric_card(
            metrics_container, "🎯 Learn Rate", config_info.get('learning_rate', 'N/A'), "#3F51B5"
        )
        
        # Threads card
        self.metric_cards['threads'] = self.create_metric_card(
            metrics_container, "⚙️ Threads", config_info.get('max_threads', 'N/A'), "#795548"
        )
        
        # Loss Function card
        self.metric_cards['loss_function'] = self.create_metric_card(
            metrics_container, "📉 Loss", config_info.get('loss_function', 'N/A'), "#E91E63"
        )
        
        # Progress card
        self.metric_cards['progress'] = self.create_progress_card(
            metrics_container, "📈 Overall Progress", 0
        )
    
    def create_metric_card(self, parent, title, value, color):
        """Create a professional metric card"""
        card_frame = tk.Frame(parent, bg='white', relief='flat', bd=0)
        card_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=5)
        
        # Add subtle shadow effect
        shadow_frame = tk.Frame(parent, bg='#e8e8e8', height=2)
        shadow_frame.pack(side=tk.LEFT, fill=tk.X, padx=8)
        
        # Color accent bar
        accent_bar = tk.Frame(card_frame, bg=color, height=3)
        accent_bar.pack(fill=tk.X)
        
        # Card content
        content_frame = tk.Frame(card_frame, bg='white')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        
        # Title
        title_label = tk.Label(
            content_frame, text=title,
            font=('Segoe UI', 8), fg='#666666', bg='white', 
            wraplength=100  # Allow text wrapping if needed
        )
        title_label.pack(anchor='w')
        
        # Value
        value_label = tk.Label(
            content_frame, text=value,
            font=('Segoe UI', 12, 'bold'), fg=color, bg='white',
            wraplength=100  # Allow text wrapping if needed
        )
        value_label.pack(anchor='w')
        
        return {'frame': card_frame, 'value_label': value_label}
    
    def create_progress_card(self, parent, title, progress):
        """Create a progress card with progress bar"""
        card_frame = tk.Frame(parent, bg='white', relief='flat', bd=0)
        card_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=5)
        
        # Color accent bar
        accent_bar = tk.Frame(card_frame, bg='#2196F3', height=3)
        accent_bar.pack(fill=tk.X)
        
        # Card content
        content_frame = tk.Frame(card_frame, bg='white')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        
        # Title
        title_label = tk.Label(
            content_frame, text=title,
            font=('Segoe UI', 8), fg='#666666', bg='white',
            wraplength=100
        )
        title_label.pack(anchor='w')
        
        # Progress bar container
        progress_container = tk.Frame(content_frame, bg='#f0f0f0', height=6)
        progress_container.pack(fill=tk.X, pady=(4, 2))
        progress_container.pack_propagate(False)
        
        # Progress bar (using place for width control)
        progress_bar = tk.Frame(progress_container, bg='#2196F3', height=6)
        progress_bar.place(x=0, y=0, width=0, height=6)  # Start with 0 width
        
        # Progress text
        progress_text = tk.Label(
            content_frame, text=f"{progress}%",
            font=('Segoe UI', 10, 'bold'), fg='#2196F3', bg='white',
            wraplength=100
        )
        progress_text.pack(anchor='w')
        
        return {
            'frame': card_frame, 
            'progress_bar': progress_bar, 
            'progress_text': progress_text,
            'progress_container': progress_container
        }
    
    def draw_neural_network_logo(self, canvas):
        """Draw a beautiful neural network logo"""
        # Logo dimensions
        width, height = 80, 80
        
        # Colors for the logo
        node_color = '#FFFFFF'
        connection_color = '#E3F2FD'
        highlight_color = '#FFF9C4'
        
        # Define node positions for a 3-layer network (Input-Hidden-Output)
        # Input layer (3 nodes)
        input_nodes = [
            (15, 20), (15, 40), (15, 60)
        ]
        
        # Hidden layer (4 nodes)
        hidden_nodes = [
            (40, 15), (40, 30), (40, 50), (40, 65)
        ]
        
        # Output layer (2 nodes)
        output_nodes = [
            (65, 30), (65, 50)
        ]
        
        # Draw connections first (so they appear behind nodes)
        canvas.create_line(0, 0, 0, 0, width=1)  # Dummy line to ensure canvas is ready
        
        # Input to Hidden connections
        for input_pos in input_nodes:
            for hidden_pos in hidden_nodes:
                canvas.create_line(
                    input_pos[0] + 4, input_pos[1], 
                    hidden_pos[0] - 4, hidden_pos[1],
                    fill=connection_color, width=1, smooth=True
                )
        
        # Hidden to Output connections
        for hidden_pos in hidden_nodes:
            for output_pos in output_nodes:
                canvas.create_line(
                    hidden_pos[0] + 4, hidden_pos[1], 
                    output_pos[0] - 4, output_pos[1],
                    fill=connection_color, width=1, smooth=True
                )
        
        # Draw nodes and store their IDs
        node_radius = 4
        self.logo_node_objects = []  # Reset list
        
        # Input layer nodes
        for i, pos in enumerate(input_nodes):
            color = highlight_color if i == 1 else node_color  # Highlight middle node
            node_id = canvas.create_oval(
                pos[0] - node_radius, pos[1] - node_radius,
                pos[0] + node_radius, pos[1] + node_radius,
                fill=color, outline='#1976D2', width=1,
                tags=f"node_input_{i}"
            )
            self.logo_node_objects.append(('input', i, node_id, pos))
        
        # Hidden layer nodes
        for i, pos in enumerate(hidden_nodes):
            color = highlight_color if i in [1, 2] else node_color  # Highlight active nodes
            node_id = canvas.create_oval(
                pos[0] - node_radius, pos[1] - node_radius,
                pos[0] + node_radius, pos[1] + node_radius,
                fill=color, outline='#1976D2', width=1,
                tags=f"node_hidden_{i}"
            )
            self.logo_node_objects.append(('hidden', i, node_id, pos))
        
        # Output layer nodes
        for i, pos in enumerate(output_nodes):
            color = highlight_color if i == 0 else node_color  # Highlight main output
            node_id = canvas.create_oval(
                pos[0] - node_radius, pos[1] - node_radius,
                pos[0] + node_radius, pos[1] + node_radius,
                fill=color, outline='#1976D2', width=2,
                tags=f"node_output_{i}"
            )
            self.logo_node_objects.append(('output', i, node_id, pos))
        
        # Add some animated-looking data flow (small moving dots)
        # These will be static but give the impression of data flow
        flow_dots = [
            (25, 35), (32, 42), (48, 35), (55, 42)
        ]
        
        for dot_pos in flow_dots:
            canvas.create_oval(
                dot_pos[0] - 1, dot_pos[1] - 1,
                dot_pos[0] + 1, dot_pos[1] + 1,
                fill='#FFD54F', outline='#FFD54F'
            )
        
        # Add a subtle frame around the logo
        canvas.create_rectangle(
            2, 2, width-2, height-2,
            outline='#BBDEFB', width=1
        )
        
        # Store canvas reference for potential animation
        self.logo_canvas = canvas
        
        # Store node information for hover effects
        self.logo_nodes = {
            'input': input_nodes,
            'hidden': hidden_nodes,
            'output': output_nodes
        }
        self.logo_node_objects = []  # Store canvas object IDs
        
        # Create invisible hover areas for each node
        self.create_hover_areas(canvas)
        
        # Logo animation removed - static logo only
    
    def create_hover_areas(self, canvas):
        """Create invisible hover areas around each node"""
        hover_radius = 8  # Larger than node radius for easier hovering
        
        # Bind mouse events to the canvas
        canvas.bind('<Motion>', self.on_logo_mouse_move)
        canvas.bind('<Leave>', self.on_logo_mouse_leave)
        
        # Store current hovered node
        self.hovered_node = None
        self.tooltip_window = None
    
    def on_logo_mouse_move(self, event):
        """Handle mouse movement over the logo"""
        x, y = event.x, event.y
        hover_radius = 10
        
        # Check which node (if any) is being hovered
        hovered_node = None
        for layer, index, node_id, pos in self.logo_node_objects:
            distance = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if distance <= hover_radius:
                hovered_node = (layer, index, node_id, pos)
                break
        
        # If hover state changed
        if hovered_node != self.hovered_node:
            # Remove previous hover effect
            if self.hovered_node:
                self.remove_hover_effect(self.hovered_node)
            
            # Add new hover effect
            if hovered_node:
                self.add_hover_effect(hovered_node)
            
            self.hovered_node = hovered_node
    
    def on_logo_mouse_leave(self, event):
        """Handle mouse leaving the logo area"""
        if self.hovered_node:
            self.remove_hover_effect(self.hovered_node)
            self.hovered_node = None
    
    def add_hover_effect(self, node_info):
        """Add glowing effect to a node"""
        layer, index, node_id, pos = node_info
        
        # Create a glowing ring around the node
        glow_radius = 7
        glow_id = self.logo_canvas.create_oval(
            pos[0] - glow_radius, pos[1] - glow_radius,
            pos[0] + glow_radius, pos[1] + glow_radius,
            fill='', outline='#FFD700', width=2,
            tags=f"glow_{layer}_{index}"
        )
        
        # Change node color to bright highlight
        self.logo_canvas.itemconfig(node_id, fill='#FFEB3B', outline='#FF6F00', width=2)
        
        # Store glow ID for removal
        self.current_glow_id = glow_id
        
        # Show tooltip with node information
        self.show_node_tooltip(pos, layer, index)
    
    def remove_hover_effect(self, node_info):
        """Remove glowing effect from a node"""
        layer, index, node_id, pos = node_info
        
        # Remove glow ring
        if hasattr(self, 'current_glow_id'):
            try:
                self.logo_canvas.delete(self.current_glow_id)
            except tk.TclError:
                pass
        
        # Restore original node color
        if layer == 'input':
            original_color = '#FFF9C4' if index == 1 else '#FFFFFF'
        elif layer == 'hidden':
            original_color = '#FFF9C4' if index in [1, 2] else '#FFFFFF'
        else:  # output
            original_color = '#FFF9C4' if index == 0 else '#FFFFFF'
        
        original_width = 2 if layer == 'output' else 1
        self.logo_canvas.itemconfig(node_id, fill=original_color, outline='#1976D2', width=original_width)
        
        # Hide tooltip
        self.hide_node_tooltip()
    
    def show_node_tooltip(self, pos, layer, index):
        """Show tooltip with node information"""
        # Define tooltip text based on layer and node
        tooltips = {
            'input': [
                "Input Neuron 1\nReceives feature data",
                "Input Neuron 2\nActive processing",
                "Input Neuron 3\nReceives feature data"
            ],
            'hidden': [
                "Hidden Neuron 1\nProcessing patterns",
                "Hidden Neuron 2\nActive learning",
                "Hidden Neuron 3\nActive learning", 
                "Hidden Neuron 4\nProcessing patterns"
            ],
            'output': [
                "Output Neuron 1\nMain prediction",
                "Output Neuron 2\nSecondary output"
            ]
        }
        
        tooltip_text = tooltips[layer][index]
        
        # Create tooltip window
        if hasattr(self, 'tooltip_window') and self.tooltip_window:
            self.tooltip_window.destroy()
        
        self.tooltip_window = tk.Toplevel(self.root)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.configure(bg='#2C2C2C')
        
        # Position tooltip relative to canvas
        canvas_x = self.logo_canvas.winfo_rootx()
        canvas_y = self.logo_canvas.winfo_rooty()
        tooltip_x = canvas_x + pos[0] + 15
        tooltip_y = canvas_y + pos[1] - 10
        
        self.tooltip_window.geometry(f"+{tooltip_x}+{tooltip_y}")
        
        # Create tooltip label
        tooltip_label = tk.Label(
            self.tooltip_window,
            text=tooltip_text,
            bg='#2C2C2C',
            fg='#FFFFFF',
            font=('Segoe UI', 9),
            relief='solid',
            borderwidth=1,
            padx=8,
            pady=4
        )
        tooltip_label.pack()
        
        # Auto-hide after 3 seconds
        self.root.after(3000, self.hide_node_tooltip)
    
    def hide_node_tooltip(self):
        """Hide the node tooltip"""
        if hasattr(self, 'tooltip_window') and self.tooltip_window:
            try:
                self.tooltip_window.destroy()
                self.tooltip_window = None
            except tk.TclError:
                pass
    
    def animate_logo(self):
        """Enhanced professional animation for the logo"""
        if not hasattr(self, 'logo_animation_state'):
            self.logo_animation_state = 0
            self.flow_dot_objects = []
            self.connection_objects = []
        
        try:
            # Clear previous flow dots
            for obj in self.flow_dot_objects:
                try:
                    self.logo_canvas.delete(obj)
                except tk.TclError:
                    pass
            self.flow_dot_objects.clear()
            
            # Enhanced flowing data animation
            flow_positions = [
                # Input to Hidden connections
                (20, 30), (25, 35), (30, 40),  # Data flow 1
                (22, 45), (28, 48), (34, 52),  # Data flow 2
                # Hidden to Output connections  
                (45, 25), (50, 28), (55, 32),  # Output flow 1
                (47, 55), (52, 52), (58, 48),  # Output flow 2
            ]
            
            # Create smooth flowing animation
            time_factor = self.logo_animation_state * 0.08
            
            for i, pos in enumerate(flow_positions):
                # Calculate wave effect for each data point
                phase = i * 0.8 + time_factor
                intensity = (math.sin(phase) + 1) / 2  # 0 to 1
                
                # Professional color gradient based on intensity
                if intensity > 0.7:
                    color = '#1976D2'  # Strong blue
                    size = 2
                elif intensity > 0.4:
                    color = '#42A5F5'  # Medium blue
                    size = 1.5
                else:
                    color = '#90CAF9'  # Light blue
                    size = 1
                
                # Only show dots above threshold for cleaner look
                if intensity > 0.3:
                    dot_obj = self.logo_canvas.create_oval(
                        pos[0] - size, pos[1] - size,
                        pos[0] + size, pos[1] + size,
                        fill=color, outline=color, width=0
                    )
                    self.flow_dot_objects.append(dot_obj)
            
            # Add subtle node pulsation for active nodes
            self.animate_active_nodes()
            
            self.logo_animation_state += 1
            
            # Smooth 60fps animation
            self.root.after(50, self.animate_logo)
            
        except (AttributeError, tk.TclError):
            pass
    
    def animate_active_nodes(self):
        """Add subtle pulsation to active nodes"""
        if not hasattr(self, 'active_node_cycle'):
            self.active_node_cycle = 0
        
        # Gentle pulsation for highlighted nodes
        pulse_intensity = (math.sin(self.active_node_cycle * 0.05) + 1) / 2
        glow_alpha = int(pulse_intensity * 40 + 15)  # 15-55 range
        
        # Professional glow color
        glow_color = f'#{hex(200 + int(pulse_intensity * 55))[2:]:0>2}' + \
                    f'{hex(230 + int(pulse_intensity * 25))[2:]:0>2}' + \
                    f'{hex(255)[2:]:0>2}'  # Light blue glow
        
        self.active_node_cycle += 1
    
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
        
    def add_data_point(self, model_id, epoch, r2_scores, training_fidelity=None, validation_fidelity=None, mean_loss=None):
        """Add a data point and update the table only when best epoch changes"""
        self.model_data[model_id]['current_epoch'] = epoch
        
        # Handle both single value and list of values for R2 scores
        if isinstance(r2_scores, (list, tuple)):
            current_r2 = list(r2_scores)
        else:
            # Single value - put it in the first position, pad others with 0
            current_r2 = [r2_scores] + [0.0] * (len(self.model_data[model_id]['best_r2_scores']) - 1)
        
        # Calculate current average R² for this epoch
        current_avg_r2 = sum(current_r2) / len(current_r2) if current_r2 else 0
        
        # Only update display data if this is a new best epoch (higher avg R²)
        is_new_best = False
        if current_avg_r2 > self.model_data[model_id]['best_avg_r2']:
            is_new_best = True
            self.model_data[model_id]['best_epoch'] = epoch
            self.model_data[model_id]['best_avg_r2'] = current_avg_r2
            self.model_data[model_id]['best_r2_scores'] = current_r2.copy()
            
            # Update best training metrics if provided
            if training_fidelity is not None:
                self.model_data[model_id]['best_training_fidelity'] = training_fidelity
            if validation_fidelity is not None:
                self.model_data[model_id]['best_validation_fidelity'] = validation_fidelity
            if mean_loss is not None:
                self.model_data[model_id]['best_loss'] = mean_loss
        
        # Update table only if we have a new best epoch
        if is_new_best:
            self.root.after(0, self.update_table_row, model_id)
        
        # Always update current epoch in table (but other values stay from best epoch)
        self.root.after(0, self.update_current_epoch, model_id)
        
        # Update dashboard periodically (based on ShowTestEvery to avoid excessive updates)
        show_every = getattr(self, 'show_test_every', 500)
        if epoch % show_every == 0:
            self.root.after(0, self.update_status)
    
    def update_current_epoch(self, model_id):
        """Update only the current epoch column without changing other values"""
        try:
            for item in self.tree.get_children():
                values = self.tree.item(item, 'values')
                if values and values[1] == self.model_labels.get(model_id, model_id):
                    # Update only the current epoch (3rd column, index 2)
                    updated_values = list(values)
                    updated_values[2] = str(self.model_data[model_id]['current_epoch'])
                    self.tree.item(item, values=updated_values)
                    break
        except (KeyError, AttributeError, tk.TclError):
            pass
        
    def mark_completed(self, model_id, final_results=None):
        """Mark a model as completed and store final results"""
        self.model_data[model_id]['completed'] = True
        
        # Store final training results if provided
        if final_results:
            self.model_data[model_id]['final_training_r2'] = final_results.get('training_fidelity', 0.0)
            self.model_data[model_id]['final_validation_r2'] = final_results.get('validation_fidelity', 0.0)
            self.model_data[model_id]['final_loss'] = final_results.get('mean_loss', 0.0)
        
        self.root.after(0, self.update_table_row, model_id)
        # Update dashboard metrics when a model completes
        self.root.after(0, self.update_status)
    
    def mark_error(self, model_id):
        """Mark a model as failed"""
        self.model_data[model_id]['error'] = True
        self.root.after(0, self.update_table_row, model_id)
        # Update dashboard metrics when a model fails
        self.root.after(0, self.update_status)
    
    def mark_paused_status(self, model_id):
        """Mark a model as paused in the GUI"""
        self.model_data[model_id]['gui_paused'] = True
        self.root.after(0, self.update_table_row, model_id)
    
    def mark_training_status(self, model_id):
        """Mark a model as training in the GUI"""
        self.model_data[model_id]['gui_paused'] = False
        self.root.after(0, self.update_table_row, model_id)
    
    def update_table_row(self, model_id):
        """Update a specific row in the table with best epoch values"""
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
        elif data.get('gui_paused', False):
            status = 'Paused'
            tag = 'paused'
        elif data['current_epoch'] > 0:
            status = 'Training'
            tag = 'training'
        else:
            status = 'Waiting'
            tag = 'waiting'
        
        # Build values: ID, Architecture, Current Epoch, Best Epoch, Best R² scores, Summary stats, Status
        model_index = data.get('id', '?')
        values = [
            str(model_index), 
            arch_label, 
            f"{data['current_epoch']}", 
            f"{data['best_epoch']}"
        ]
        
        # Add best R² scores for each output variable
        for i, r2_score in enumerate(data['best_r2_scores']):
            values.append(f"{r2_score:.2f}")
        
        # Add summary statistics (only average R² if multiple outputs)
        if len(self.dataset_info['output_variables']) > 1:
            values.append(f"{data['best_avg_r2']:.2f}")
        
        # Add best metrics from best epoch
        values.extend([
            f"{data['best_training_fidelity']:.2f}",
            f"{data['best_validation_fidelity']:.2f}",
            f"{data['best_loss']:.2e}" if data['best_loss'] is not None else "N/A"
        ])
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
        """Update the status bar and dashboard"""
        # Count only non-deleted models
        active_models = {k: v for k, v in self.model_data.items() if k not in self.deleted_models}
        
        completed = len([m for m in active_models.values() if m['completed']])
        training = len([m for m in active_models.values() if m['current_epoch'] > 0 and not m['completed'] and not m['error']])
        waiting = len([m for m in active_models.values() if m['current_epoch'] == 0 and not m['error']])
        errors = len([m for m in active_models.values() if m['error']])
        
        # Total = original configurations + pending configurations (excluding deleted)
        original_configs = len(self.param_combinations)
        pending = len(getattr(self, 'pending_configurations', []))
        deleted_count = len(self.deleted_models)
        
        total = original_configs + pending - deleted_count
        
        # Update traditional status bar
        status_text = f"Total: {total} | Training: {training} | Completed: {completed} | Waiting: {waiting} | Errors: {errors}"
        if hasattr(self, 'status_label'):
            self.status_label.config(text=status_text)
        
        # Update dashboard metrics
        self.update_dashboard_metrics(total, training, completed, waiting, errors)
        
        # Update running indicator with professional icons
        if training > 0:
            self.status_indicator.config(text="● Running", fg='#4caf50')
        elif completed + errors >= total and total > 0 and pending == 0:
            self.status_indicator.config(text="✓ Completed", fg='#4caf50')
        elif waiting > 0 or pending > 0:
            self.status_indicator.config(text="○ Waiting", fg='#ff9800')
        else:
            self.status_indicator.config(text="- Idle", fg='#757575')
    
    def update_dashboard_metrics(self, total, training, completed, waiting, errors):
        """Update dashboard metric cards"""
        if not hasattr(self, 'metric_cards'):
            return
        
        try:
            # Update metric values
            self.metric_cards['total']['value_label'].config(text=str(total))
            self.metric_cards['training']['value_label'].config(text=str(training))
            self.metric_cards['completed']['value_label'].config(text=str(completed))
            
            # Calculate best average R² from all models (completed and training)
            best_avg_r2 = 0.0
            all_models = list(self.model_data.values())
            if all_models:
                avg_r2_values = []
                for model in all_models:
                    # Use best_avg_r2 if available, otherwise calculate from current r2_scores
                    if 'best_avg_r2' in model and model['best_avg_r2'] > 0:
                        avg_r2_values.append(model['best_avg_r2'])
                    elif 'best_r2_scores' in model and model['best_r2_scores']:
                        current_avg = sum(model['best_r2_scores']) / len(model['best_r2_scores'])
                        avg_r2_values.append(current_avg)
                
                if avg_r2_values:
                    best_avg_r2 = max(avg_r2_values)
            
            self.metric_cards['best_r2']['value_label'].config(text=f"{best_avg_r2:.2f}%")
            
            # Update progress bar
            if total > 0:
                progress = int((completed / total) * 100)
                self.update_progress_bar(progress)
            
        except (KeyError, AttributeError, tk.TclError):
            pass
    
    def update_progress_bar(self, progress):
        """Update the progress bar"""
        try:
            progress_card = self.metric_cards['progress']
            
            # Update progress text
            progress_card['progress_text'].config(text=f"{progress}%")
            
            # Update progress bar width using place geometry manager
            progress_container = progress_card['progress_container']
            progress_bar = progress_card['progress_bar']
            
            # Calculate the width as percentage of container
            container_width = progress_container.winfo_width()
            if container_width <= 1:  # Widget not yet rendered
                container_width = 100  # Use default width
            
            bar_width = int((progress / 100) * container_width)
            
            # Use place to set exact width
            progress_bar.place(x=0, y=0, width=bar_width, height=6)
                
        except (KeyError, AttributeError, tk.TclError):
            pass
        
    def collect_training_data(self, data_queue):
        """Collect data from training processes with optimized polling"""
        batch_updates = []
        last_update_time = time.time()
        
        while self.data_collection_active:
            try:
                # Use short timeout for responsiveness, batch updates for efficiency
                data = data_queue.get(timeout=0.1)
                if data is None:
                    break
                
                # Queue the update instead of processing immediately
                batch_updates.append(data)
                
                # Process updates in batches every 0.5 seconds or when batch gets large
                current_time = time.time()
                if len(batch_updates) >= 10 or (current_time - last_update_time) >= 0.5:
                    # Schedule batch update on main thread using after_idle for better responsiveness
                    self.root.after_idle(lambda updates=batch_updates.copy(): self.process_batch_updates(updates))
                    batch_updates.clear()
                    last_update_time = current_time
                
            except Empty:
                # Process any remaining updates when queue is empty
                if batch_updates:
                    current_time = time.time()
                    if (current_time - last_update_time) >= 0.2:  # Shorter delay when queue is empty
                        self.root.after_idle(lambda updates=batch_updates.copy(): self.process_batch_updates(updates))
                        batch_updates.clear()
                        last_update_time = current_time
                
                # Small sleep to prevent busy waiting and improve GUI responsiveness
                time.sleep(0.05)
                continue
                
            except Exception as e:
                # Handle other exceptions with logging, but only if GUI is still active
                if self.data_collection_active:
                    print(f"Error in data collection: {e}")
                continue
    
    def process_batch_updates(self, updates):
        """Process a batch of updates on the main GUI thread"""
        for data in updates:
            try:
                if len(data) == 6:
                    # Enhanced training data with live metrics
                    model_id, epoch, r2_scores, training_fidelity, validation_fidelity, mean_loss = data
                    self.add_data_point(model_id, epoch, r2_scores, training_fidelity, validation_fidelity, mean_loss)
                elif len(data) == 3 and isinstance(data[1], int):
                    # Legacy training data format (backward compatibility) - epoch is integer
                    model_id, epoch, r2_scores = data
                    self.add_data_point(model_id, epoch, r2_scores)
                elif len(data) >= 2 and data[1] == 'COMPLETED':
                    # Completion marker with optional final results
                    model_id = data[0]
                    final_results = data[2] if len(data) > 2 else None
                    self.mark_completed(model_id, final_results)
                elif len(data) == 2 and data[1] == 'ERROR':
                    # Error marker
                    model_id = data[0]
                    self.mark_error(model_id)
                elif len(data) == 2 and data[1] == 'PAUSED':
                    # Pause marker
                    model_id = data[0]
                    self.mark_paused_status(model_id)
                elif len(data) == 2 and data[1] == 'TRAINING':
                    # Resume training marker
                    model_id = data[0]
                    self.mark_training_status(model_id)
            except Exception as e:
                print(f"Error processing update {data}: {e}")
                continue
    
    def stop_data_collection(self):
        """Stop the data collection thread"""
        self.data_collection_active = False
                
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
        import subprocess
        import platform
        print("\nGUI closing - terminating all training processes...")
        
        # Set global shutdown flag to stop training threads
        try:
            # Import the shutdown flag from main module
            import __main__
            if hasattr(__main__, 'shutdown_requested'):
                __main__.shutdown_requested = True
                print("[SHUTDOWN] Global shutdown flag set")
        except:
            pass
        
        # Force terminate the entire process tree
        try:
            # First try graceful close
            self.root.quit()
            self.root.destroy()
            
            # Then force terminate entire process tree on Windows
            if platform.system() == "Windows":
                pid = os.getpid()
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                             timeout=1, capture_output=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        
        # Find column index - updated structure: ID, Architecture, Epoch, Best_Epoch, R2s, Avg_R2?, Training, Validation, Loss, Status
        columns = ['ID', 'Architecture', 'Epoch', 'Best_Epoch'] + [f'{var}_R2' for var in self.dataset_info['output_variables']]
        if len(self.dataset_info['output_variables']) > 1:
            columns.extend(['Avg_R2', 'Training_R2', 'Validation_R2', 'Loss'])
        else:
            columns.extend(['Training_R2', 'Validation_R2', 'Loss'])
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
            elif col_name in ['Epoch', 'Best_Epoch']:
                return int(value) if value.isdigit() else 0
            elif '_R2' in col_name or 'Avg_R2' in col_name or col_name in ['Training_R2', 'Validation_R2']:
                try:
                    return float(value)
                except ValueError:
                    return 0.0
            elif col_name == 'Loss':
                if value == "N/A":
                    return float('inf')  # N/A goes to the end when sorting ascending
                try:
                    return float(value)
                except ValueError:
                    return float('inf')
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
    
    def on_tree_click(self, event):
        """Handle tree click for deselection functionality"""
        # Get the item that was clicked
        item = self.tree.identify_row(event.y)
        
        # If clicking on the same item that's already selected, deselect it
        if item and item == self.last_selected_item:
            self.tree.selection_remove(item)
            self.last_selected_item = None
        else:
            # Update the last selected item
            self.last_selected_item = item
    
    def on_tree_hover(self, event):
        """Handle hover effects on table rows"""
        item = self.tree.identify_row(event.y)
        # Simple hover tracking without visual changes to avoid Tkinter errors
        self.last_hovered_item = item
    
    def on_tree_leave(self, event):
        """Handle leaving table area"""
        self.last_hovered_item = None
    
    def on_right_click(self, event):
        """Handle right click for context menu - supports multi-select"""
        # Get the item that was clicked
        item = self.tree.identify_row(event.y)
        if not item:
            return
        
        # Get currently selected items
        selected_items = self.tree.selection()
        
        # If the clicked item is not in selection, select only it
        if item not in selected_items:
            self.tree.selection_set(item)
            selected_items = [item]
        
        # Get model info for selected items
        selected_models = []
        for selected_item in selected_items:
            values = self.tree.item(selected_item, 'values')
            if not values:
                continue
                
            model_id = None
            # Find model_id by matching architecture
            arch_text = values[1]  # Architecture column
            for mid, label in self.model_labels.items():
                if label == arch_text:
                    model_id = mid
                    break
            
            if model_id:
                selected_models.append((model_id, values))
        
        if not selected_models:
            return
        
        # Show appropriate context menu based on selection count
        if len(selected_models) == 1:
            self.show_context_menu(event, selected_models[0][0], selected_models[0][1])
        else:
            self.show_batch_context_menu(event, selected_models)
    
    def show_context_menu(self, event, model_id, values):
        """Show context menu for model management"""
        context_menu = tk.Menu(self.root, tearoff=0, bg='white', fg='black')
        
        status = values[-1]  # Last column is status
        
        # Delete option (always available except for completed)
        if status not in ['Completed']:
            context_menu.add_command(
                label="🗑️ Delete Configuration",
                command=lambda: self.delete_model(model_id)
            )
            context_menu.add_separator()
        
        
        # Pause/Resume options
        if status == 'Training':
            if model_id in self.paused_models:
                context_menu.add_command(
                    label="▶️ Resume Training",
                    command=lambda: self.resume_model(model_id)
                )
            else:
                context_menu.add_command(
                    label="⏸️ Pause Training",
                    command=lambda: self.pause_model(model_id)
                )
        elif status == 'Paused':
            context_menu.add_command(
                label="▶️ Resume Training",
                command=lambda: self.resume_model(model_id)
            )
        elif status == 'Waiting' and model_id in self.paused_models:
            context_menu.add_command(
                label="▶️ Resume (Remove from Pause)",
                command=lambda: self.resume_model(model_id)
            )
        elif status in ['Waiting']:
            context_menu.add_command(
                label="⏸️ Pause (Prevent Start)",
                command=lambda: self.pause_model(model_id)
            )
        
        # Info option
        context_menu.add_separator()
        context_menu.add_command(
            label="ℹ️ Model Info",
            command=lambda: self.show_model_info(model_id, values)
        )
        
        # Show menu at cursor position
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def show_batch_context_menu(self, event, selected_models):
        """Show context menu for batch operations on multiple models"""
        context_menu = tk.Menu(self.root, tearoff=0, bg='white', fg='black')
        
        # Get status summary
        statuses = [values[-1] for _, values in selected_models]
        model_ids = [model_id for model_id, _ in selected_models]
        
        # Count different statuses
        training_count = sum(1 for s in statuses if s == 'Training')
        paused_count = sum(1 for s in statuses if s == 'Paused')
        waiting_count = sum(1 for s in statuses if s == 'Waiting')
        completed_count = sum(1 for s in statuses if s == 'Completed')
        
        # Batch delete (always available except for completed)
        deletable_count = len([s for s in statuses if s not in ['Completed']])
        if deletable_count > 0:
            context_menu.add_command(
                label=f"🗑️ Delete {deletable_count} Configurations",
                command=lambda: self.batch_delete_models(model_ids)
            )
            context_menu.add_separator()
        
        # Batch pause (for training and waiting models)
        pausable_count = len([s for s in statuses if s in ['Training', 'Waiting']])
        if pausable_count > 0:
            context_menu.add_command(
                label=f"⏸️ Pause {pausable_count} Models",
                command=lambda: self.batch_pause_models(model_ids)
            )
        
        # Batch resume (for models that are individually paused)
        resumable_count = len([mid for mid in model_ids if mid in self.paused_models])
        if resumable_count > 0:
            context_menu.add_command(
                label=f"▶️ Resume {resumable_count} Models",
                command=lambda: self.batch_resume_models(model_ids)
            )
        
        # Batch info
        context_menu.add_separator()
        context_menu.add_command(
            label=f"ℹ️ Info for {len(selected_models)} Models",
            command=lambda: self.show_batch_info(selected_models)
        )
        
        # Show menu at cursor position
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def batch_delete_models(self, model_ids):
        """Delete multiple model configurations"""
        from tkinter import messagebox
        
        # Filter out completed models
        deletable_models = [mid for mid in model_ids 
                           if mid in self.model_data and 
                           not self.model_data[mid].get('completed', False)]
        
        if not deletable_models:
            messagebox.showinfo("No Deletable Models", "No models can be deleted (all are completed)")
            return
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Batch Deletion", 
                              f"Are you sure you want to delete {len(deletable_models)} configurations?\n\n"
                              f"Models: {', '.join(deletable_models)}\n\n"
                              "This action cannot be undone."):
            
            deleted_count = 0
            for model_id in deletable_models:
                # Mark as deleted
                self.deleted_models.add(model_id)
                
                # Remove from GUI table
                if model_id in self.table_items:
                    item = self.table_items[model_id]
                    self.tree.delete(item)
                    del self.table_items[model_id]
                    deleted_count += 1
                
                # Remove from model data
                if model_id in self.model_data:
                    del self.model_data[model_id]
                
                # Remove from other tracking lists
                self.paused_models.discard(model_id)
            
            print(f"[GUI] Batch deleted {deleted_count} configurations: {deletable_models}")
            self.update_status()
    
    def batch_pause_models(self, model_ids):
        """Pause multiple model configurations"""
        paused_count = 0
        for model_id in model_ids:
            # Only pause if not completed or already paused
            if (model_id in self.model_data and 
                not self.model_data[model_id].get('completed', False) and
                model_id not in self.paused_models):
                
                self.paused_models.add(model_id)
                # Update external pause state for training threads
                if self.external_pause_state:
                    self.external_pause_state["paused"].add(model_id)
                paused_count += 1
                
                # Update visual status
                if model_id in self.table_items:
                    item = self.table_items[model_id]
                    current_values = list(self.tree.item(item, 'values'))
                    current_values[-1] = 'Paused'
                    self.tree.item(item, values=tuple(current_values), tags=('paused',))
        
        if paused_count > 0:
            print(f"[GUI] Batch paused {paused_count} models: {[mid for mid in model_ids if mid in self.paused_models]}")
            # Configure paused tag
            self.tree.tag_configure('paused', foreground='#ff9800', 
                                   font=('Segoe UI', 10, 'italic'))
    
    def batch_resume_models(self, model_ids):
        """Resume multiple model configurations"""
        resumed_count = 0
        for model_id in model_ids:
            if model_id in self.paused_models:
                self.paused_models.discard(model_id)
                # Update external pause state for training threads
                if self.external_pause_state:
                    self.external_pause_state["paused"].discard(model_id)
                resumed_count += 1
                
                # Update visual status back to appropriate state
                data = self.model_data.get(model_id, {})
                if data.get('epoch', 0) > 0:
                    status = 'Training'
                    tag = 'training'
                else:
                    status = 'Waiting'
                    tag = 'waiting'
                
                if model_id in self.table_items:
                    item = self.table_items[model_id]
                    current_values = list(self.tree.item(item, 'values'))
                    current_values[-1] = status
                    self.tree.item(item, values=tuple(current_values), tags=(tag,))
        
        if resumed_count > 0:
            print(f"[GUI] Batch resumed {resumed_count} models")
    
    def show_batch_info(self, selected_models):
        """Show summary information for multiple models"""
        from tkinter import messagebox
        
        # Prepare summary info
        info_text = f"Batch Information for {len(selected_models)} Models\n\n"
        
        # Status summary
        statuses = [values[-1] for _, values in selected_models]
        status_counts = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        info_text += "Status Summary:\n"
        for status, count in status_counts.items():
            info_text += f"  {status}: {count}\n"
        
        info_text += "\nModel List:\n"
        for model_id, values in selected_models[:10]:  # Show first 10
            info_text += f"  {model_id} ({values[1]}) - {values[-1]}\n"
        
        if len(selected_models) > 10:
            info_text += f"  ... and {len(selected_models) - 10} more models\n"
        
        # Performance summary for completed models
        completed_models = [(mid, values) for mid, values in selected_models if values[-1] == 'Completed']
        if completed_models:
            info_text += "\nPerformance Summary (Completed Models):\n"
            # Calculate the correct column index for Max R² data
            # Columns: ID, Architecture, Epoch, [Output_R2 columns], Summary_R2, Best_Epoch, Optimal_Stop, Status
            r2_col_idx = 3 + len(self.dataset_info['output_variables'])  # Summary R² column (Max_R2 or Avg_Max_R2)
            
            try:
                r2_scores = [float(values[r2_col_idx]) for _, values in completed_models if len(values) > r2_col_idx]
                if r2_scores:
                    avg_r2 = sum(r2_scores) / len(r2_scores)
                    max_r2 = max(r2_scores)
                    min_r2 = min(r2_scores)
                    info_text += f"  Average R²: {avg_r2:.2f}%\n"
                    info_text += f"  Best R²: {max_r2:.2f}%\n"
                    info_text += f"  Worst R²: {min_r2:.2f}%\n"
            except (ValueError, IndexError):
                info_text += "  Performance data not available\n"
        
        messagebox.showinfo(f"Batch Info - {len(selected_models)} Models", info_text)
    
    def delete_model(self, model_id):
        """Delete a model configuration"""
        from tkinter import messagebox
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Deletion", 
                              f"Are you sure you want to delete configuration {model_id}?\n\n"
                              "This action cannot be undone."):
            
            # Mark as deleted
            self.deleted_models.add(model_id)
            
            # Remove from GUI table
            if model_id in self.table_items:
                item = self.table_items[model_id]
                self.tree.delete(item)
                del self.table_items[model_id]
            
            # Remove from model data
            if model_id in self.model_data:
                del self.model_data[model_id]
            
            # Remove from other tracking lists
            self.paused_models.discard(model_id)
            
            print(f"[GUI] Deleted configuration {model_id}")
            self.update_status()
    
    
    def pause_model(self, model_id):
        """Pause individual model training"""
        self.paused_models.add(model_id)
        # Update external pause state for training threads
        if self.external_pause_state:
            self.external_pause_state["paused"].add(model_id)
        print(f"[GUI] Paused training for model {model_id}")
        
        # Update visual status
        if model_id in self.table_items:
            item = self.table_items[model_id]
            current_values = list(self.tree.item(item, 'values'))
            current_values[-1] = 'Paused'
            self.tree.item(item, values=tuple(current_values), tags=('paused',))
        
        # Configure paused tag
        self.tree.tag_configure('paused', foreground='#ff9800', 
                               font=('Segoe UI', 10, 'italic'))
    
    def resume_model(self, model_id):
        """Resume individual model training"""
        self.paused_models.discard(model_id)
        # Update external pause state for training threads
        if self.external_pause_state:
            self.external_pause_state["paused"].discard(model_id)
        print(f"[GUI] Resumed training for model {model_id}")
        
        # Update visual status back to appropriate state
        data = self.model_data.get(model_id, {})
        if data.get('epoch', 0) > 0:
            status = 'Training'
            tag = 'training'
        else:
            status = 'Waiting'
            tag = 'waiting'
        
        if model_id in self.table_items:
            item = self.table_items[model_id]
            current_values = list(self.tree.item(item, 'values'))
            current_values[-1] = status
            self.tree.item(item, values=tuple(current_values), tags=(tag,))
    
    def show_model_info(self, model_id, values):
        """Show detailed model information"""
        from tkinter import messagebox
        
        data = self.model_data.get(model_id, {})
        current_epoch = data.get('current_epoch', 0)
        max_epochs = self.config.get('NeuralNetworkModel', {}).get('epochs', 0) if hasattr(self, 'config') else 10000
        
        info_text = f"Model Configuration: {model_id}\n\n"
        info_text += f"Architecture: {values[1]}\n"
        info_text += f"Current Epoch: {values[2]}\n"
        info_text += f"Status: {values[-1]}\n\n"
        
        # Training metrics (if available)
        if data.get('current_training_r2') is not None or data.get('current_validation_r2') is not None:
            info_text += "Training Metrics:\n"
            if data.get('current_training_r2') is not None:
                info_text += f"Training R²: {data['current_training_r2']:.2f}%\n"
            if data.get('current_validation_r2') is not None:
                info_text += f"Validation R²: {data['current_validation_r2']:.2f}%\n"
            if data.get('current_loss') is not None:
                info_text += f"Current Loss: {data['current_loss']:.2E}\n"
            info_text += "\n"
        
        # R² scores per variable
        info_text += "R² Scores by Variable:\n"
        for i, var in enumerate(self.dataset_info['output_variables']):
            if i < len(data.get('best_r2_scores', [])):
                best_r2 = data['best_r2_scores'][i]
                info_text += f"{var}: {best_r2:.2f}%\n"
        
        info_text += f"\nBest Epoch: {data.get('best_epoch', 0)}\n"
        if current_epoch > 0 and max_epochs > 0:
            progress = (current_epoch / max_epochs) * 100
            info_text += f"Progress: {progress:.1f}% ({current_epoch}/{max_epochs})\n"
        
        # Dataset info (global)
        info_text += f"\nDataset Information:\n"
        info_text += f"Dataset: {self.dataset_info.get('filename', 'Unknown')}\n"
        if hasattr(self, 'dataset_info'):
            info_text += f"Training samples: {self.dataset_info.get('train_samples', 'N/A')}"
            info_text += f", Validation: {self.dataset_info.get('val_samples', 'N/A')}\n"
            info_text += f"Input variables: {len(self.dataset_info.get('input_variables', []))}"
            info_text += f", Output variables: {len(self.dataset_info.get('output_variables', []))}\n"
        
        # Configuration info (global)
        if hasattr(self, 'config'):
            nn_config = self.config.get('NeuralNetworkModel', {})
            info_text += f"\nConfiguration:\n"
            info_text += f"Learning rate: {nn_config.get('learning_rate', 'N/A')}\n"
            info_text += f"Optimizer: {nn_config.get('UpdateMethod', 'N/A')}\n"
            info_text += f"Max epochs: {nn_config.get('epochs', 'N/A')}\n"
        
        # Status info
        if model_id in self.paused_models:
            info_text += "\n⏸️ Currently Paused"
        
        messagebox.showinfo(f"Model Info - {model_id}", info_text)
    
    def is_model_paused(self, model_id):
        """Check if a specific model is paused"""
        return model_id in self.paused_models
    
    def is_model_deleted(self, model_id):
        """Check if a specific model is deleted"""
        return model_id in self.deleted_models
    
    
    def export_table(self):
        """Export current table data to CSV file"""
        try:
            import csv
            import os
            from datetime import datetime
            from tkinter import filedialog, messagebox
            
            # Get current timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"neural_network_sweep_results_{timestamp}.csv"
            
            # Try to get desktop path for default location
            try:
                desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
                default_dir = desktop if os.path.exists(desktop) else os.getcwd()
            except:
                default_dir = os.getcwd()
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=default_dir,
                initialfile=default_filename,
                title="Export Table Data"
            )
            
            if not filename:
                return  # User cancelled
            
            # Ensure we can write to the file
            try:
                with open(filename, 'w') as test_file:
                    pass
            except Exception as e:
                messagebox.showerror("Export Error", f"Cannot write to file:\n{filename}\n\nError: {str(e)}")
                return
            
            # Get column headers
            columns = ['ID', 'Architecture', 'Epoch'] + [f'{var}_R2' for var in self.dataset_info['output_variables']]
            if len(self.dataset_info['output_variables']) > 1:
                columns.extend(['Avg_Max_R2', 'Best_Epoch', 'Training_R2', 'Validation_R2', 'Loss'])
            else:
                columns.extend(['Max_R2', 'Best_Epoch', 'Training_R2', 'Validation_R2', 'Loss'])
            columns.append('Status')
            
            # Get all table data in current display order
            table_data = []
            for item in self.tree.get_children():
                values = list(self.tree.item(item, 'values'))
                table_data.append(values)
            
            if not table_data:
                messagebox.showwarning("No Data", "No data to export. The table is empty.")
                return
            
            # Write to CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write metadata header
                writer.writerow([f'# Neural Network Architecture Sweep Results'])
                writer.writerow([f'# Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
                
                # Safe access to dataset info
                dataset_name = self.dataset_info.get('dataset_name', 'Unknown Dataset')
                dataset_filename = self.dataset_info.get('filename', 'Unknown File')
                output_vars = self.dataset_info.get('output_variables', ['Unknown Variable'])
                
                writer.writerow([f'# Dataset: {dataset_name} ({dataset_filename})'])
                writer.writerow([f'# Output Variables: {", ".join(output_vars)}'])
                writer.writerow([f'# Total Configurations: {len(table_data)}'])
                writer.writerow([])  # Empty line
                
                # Write column headers
                writer.writerow(columns)
                
                # Write data
                for row in table_data:
                    # Ensure all values are strings
                    clean_row = [str(val) if val is not None else '' for val in row]
                    writer.writerow(clean_row)
                
                # Write summary statistics
                writer.writerow([])  # Empty line
                writer.writerow(['# Summary Statistics'])
                
                completed = len([row for row in table_data if len(row) > 0 and str(row[-1]) == 'Completed'])
                training = len([row for row in table_data if len(row) > 0 and str(row[-1]) == 'Training'])
                waiting = len([row for row in table_data if len(row) > 0 and str(row[-1]) == 'Waiting'])
                errors = len([row for row in table_data if len(row) > 0 and str(row[-1]) == 'Error'])
                paused = len([row for row in table_data if len(row) > 0 and str(row[-1]) == 'Paused'])
                
                writer.writerow(['Total Configurations', len(table_data)])
                writer.writerow(['Completed', completed])
                writer.writerow(['Training', training])
                writer.writerow(['Waiting', waiting])
                writer.writerow(['Paused', paused])
                writer.writerow(['Errors', errors])
            
            # Show success message
            messagebox.showinfo("Export Successful", 
                               f"Table data exported successfully to:\n{os.path.basename(filename)}\n\n"
                               f"Exported {len(table_data)} configurations")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror("Export Error", 
                               f"Failed to export table data:\n\n{str(e)}\n\n"
                               f"Details:\n{error_details}")
    
    def toggle_pause(self):
        """Toggle pause/resume state"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_button.config(text="▶ Resume", bg='#4caf50')
            self.status_indicator.config(text="■ Paused", fg='#ff9800')
            print(f"[GUI] GLOBAL TRAINING PAUSED by user - is_paused = {self.is_paused}")
            # Mark all training models as paused visually
            self._update_global_pause_formatting(paused=True)
        else:
            self.pause_button.config(text="■ Pause", bg=self.colors['warning'])
            print(f"[GUI] GLOBAL TRAINING RESUMED by user - is_paused = {self.is_paused}")
            # Remove pause formatting from all models
            self._update_global_pause_formatting(paused=False)
        
        # Force update status display (this will set the correct indicator)
        self.update_status()
    
    def _update_global_pause_formatting(self, paused):
        """Update visual formatting for global pause state"""
        for model_id, item in self.table_items.items():
            if model_id in self.model_data:
                data = self.model_data[model_id]
                # Only affect models that are not completed and not individually paused
                if (not data.get('completed', False) and 
                    model_id not in self.paused_models and 
                    model_id not in self.deleted_models):
                    
                    current_values = list(self.tree.item(item, 'values'))
                    if paused:
                        # Mark as globally paused
                        current_values[-1] = 'Paused (Global)'
                        self.tree.item(item, values=tuple(current_values), tags=('paused',))
                    else:
                        # Restore normal status
                        if data.get('epoch', 0) > 0:
                            current_values[-1] = 'Training'
                            self.tree.item(item, values=tuple(current_values), tags=('training',))
                        else:
                            current_values[-1] = 'Waiting'
                            self.tree.item(item, values=tuple(current_values), tags=())
    
    def open_add_config_window(self):
        """Open window to add new configurations"""
        AddConfigWindow(self)
    
    def add_new_configurations(self, new_configs):
        """Add new configurations to the sweep"""
        print(f"[GUI] Checking {len(new_configs)} configurations for duplicates...")
        
        # Get all existing configurations (initial + pending + processed)
        existing_configs = set()
        
        # Add initial configurations
        for config in self.param_combinations:
            config_tuple = tuple(config)
            existing_configs.add(config_tuple)
        
        # Add pending configurations
        for config in self.pending_configurations:
            config_tuple = tuple(config)
            existing_configs.add(config_tuple)
        
        # Add configurations from model_labels (already processed)
        for model_id in self.model_labels.keys():
            # Reconstruct config from model_id
            config = self._model_id_to_config(model_id)
            if config:
                existing_configs.add(tuple(config))
        
        # Filter out duplicates
        unique_configs = []
        duplicates = []
        
        for config in new_configs:
            config_tuple = tuple(config)
            if config_tuple not in existing_configs:
                unique_configs.append(config)
                existing_configs.add(config_tuple)  # Add to prevent duplicates within new_configs
            else:
                # Convert to model_id for user-friendly message
                model_id = self._config_to_model_id(config)
                duplicates.append(model_id)
        
        # Report results
        if duplicates:
            print(f"[GUI] Skipped {len(duplicates)} duplicate configurations: {duplicates}")
        
        if not unique_configs:
            print(f"[GUI] No new configurations to add - all were duplicates")
            return
        
        print(f"[GUI] Adding {len(unique_configs)} unique configurations (skipped {len(duplicates)} duplicates)")
        
        # Add only unique configurations to pending list (for GUI display)
        self.pending_configurations.extend(unique_configs)
        
        # Send configurations to training thread via queue
        if self.new_config_queue:
            for config in unique_configs:
                self.new_config_queue.put(config)
                print(f"[INFO] Sent config to training queue")
        
        # Add to the table immediately
        for config in unique_configs:
            total_layers = config[-1]
            active_neurons = config[:-1][:total_layers]
            
            if total_layers == 1:
                model_id = f"{active_neurons[0]}"
                label_text = f"{active_neurons[0]}"
            else:
                model_id = "x".join(map(str, active_neurons))
                label_text = "x".join(map(str, active_neurons))
            
            print(f"[GUI] Adding config {model_id} to table and pending queue")
            
            # Add to model labels
            self.model_labels[model_id] = label_text
            
            # Get next available ID
            new_id = self.next_config_id
            self.next_config_id += 1  # Increment for next configuration
            
            # Store ID in model data
            self.model_data[model_id]['id'] = new_id
            
            # Create table row with correct structure
            initial_values = [str(new_id), label_text, '0', '0']  # ID, Architecture, Current Epoch, Best Epoch
            initial_values.extend(['0.00'] * len(self.dataset_info['output_variables']))  # R² scores per variable
            
            # Add Avg R² column only if multiple outputs
            if len(self.dataset_info['output_variables']) > 1:
                initial_values.append('0.00')  # Avg R²
            
            # Add remaining summary columns: Training Fidelity, Validation Fidelity, Loss
            initial_values.extend(['0.00', '0.00', 'N/A'])  # Training, Validation, Loss
            initial_values.append('Waiting')
            
            # Insert row with normal status (black text)
            item = self.tree.insert('', 'end', values=tuple(initial_values))
            self.table_items[model_id] = item
        
        
        # Set flag to force thread pool check
        self.force_check_flag = True
        print(f"[GUI] {len(unique_configs)} configurations added, total pending: {len(self.pending_configurations)}")
        
        # Update scrollbar visibility since we added new rows
        self.update_scrollbar_visibility()
        
        self.update_status()
        
        # Show user feedback about duplicates
        if duplicates:
            from tkinter import messagebox
            if len(unique_configs) > 0:
                messagebox.showinfo("Configurations Added", 
                                   f"Added {len(unique_configs)} new configurations.\\n\\n"
                                   f"Skipped {len(duplicates)} duplicates:\\n" + 
                                   ", ".join(duplicates[:10]) + 
                                   ("..." if len(duplicates) > 10 else ""))
            else:
                messagebox.showwarning("No New Configurations", 
                                      f"All {len(duplicates)} configurations already exist:\\n" + 
                                      ", ".join(duplicates[:10]) + 
                                      ("..." if len(duplicates) > 10 else ""))
    
    def _config_to_model_id(self, config):
        """Helper to convert config to model_id for debugging"""
        total_layers = config[-1]
        active_neurons = config[:-1][:total_layers]
        if total_layers == 1:
            return f"{active_neurons[0]}"
        else:
            return "x".join(map(str, active_neurons))
    
    def _model_id_to_config(self, model_id):
        """Convert model_id back to config format for duplicate checking"""
        try:
            if 'x' in model_id:
                # Multi-layer: "5x10x15" -> [5, 10, 15, 0, 0, 3]
                neurons = [int(x) for x in model_id.split('x')]
                total_layers = len(neurons)
                # Pad with zeros up to max supported layers
                max_layers = 10
                config = neurons + [0] * (max_layers - len(neurons)) + [total_layers]
                return config
            else:
                # Single layer: "10" -> [10, 0, 0, 0, 0, 1]
                neurons = int(model_id)
                max_layers = 10
                config = [neurons] + [0] * (max_layers - 1) + [1]
                return config
        except:
            return None

class AddConfigWindow:
    def __init__(self, parent_gui):
        self.parent = parent_gui
        self.window = tk.Toplevel(parent_gui.root)
        self.window.title("Add New Configurations")
        self.window.geometry("1200x700")  # Wider layout with side panel
        self.window.configure(bg='#f8f9fa')  # Modern light background
        self.window.transient(parent_gui.root)
        self.window.grab_set()
        
        # Modern colors matching main window
        self.colors = {
            'background': '#f8f9fa',
            'card': '#ffffff',
            'primary': '#64b5f6',
            'text': '#2c3e50',
            'secondary_text': '#6c757d',
            'border': '#e9ecef',
            'accent': '#28a745'
        }
        
        # Configuration list
        self.configurations = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the add configuration UI with modern design"""
        # Modern title with better styling
        title_label = tk.Label(self.window, text="Add New Architecture Configurations", 
                              font=('Segoe UI', 16, 'bold'), fg=self.colors['text'], bg=self.colors['background'])
        title_label.pack(pady=20)
        
        # Main horizontal container with modern colors
        main_container = tk.Frame(self.window, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side: Input tabs with modern styling
        left_frame = tk.Frame(main_container, bg=self.colors['background'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Create modern notebook with larger selected tab
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Modern.TNotebook', background=self.colors['background'], borderwidth=0)
        style.configure('Modern.TNotebook.Tab', 
                       background=self.colors['card'],
                       foreground=self.colors['text'],
                       padding=[25, 8],  # Normal padding for unselected tabs
                       font=('Segoe UI', 11),
                       borderwidth=1,
                       relief='solid',
                       width=18)  # Base width for tabs
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', self.colors['primary']),
                            ('active', self.colors['border'])],
                 foreground=[('selected', 'white'),
                            ('active', self.colors['text'])],
                 padding=[('selected', [35, 12]),  # Larger padding for selected tab
                         ('active', [28, 9]),      # Slightly larger for hover
                         ('!selected', [25, 8])],  # Normal for unselected
                 font=[('selected', ('Segoe UI', 11, 'bold')),  # Bold font for selected
                       ('!selected', ('Segoe UI', 11))])
        
        notebook = ttk.Notebook(left_frame, style='Modern.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Single configuration
        single_frame = tk.Frame(notebook, bg=self.colors['card'])
        notebook.add(single_frame, text="Single Configuration")
        
        # Tab 2: Bulk configuration  
        bulk_frame = tk.Frame(notebook, bg=self.colors['card'])
        notebook.add(bulk_frame, text="Bulk Configuration")
        
        self.setup_single_tab(single_frame)
        self.setup_bulk_tab(bulk_frame)
        
        # Right side: Configuration list and buttons with modern styling
        right_frame = tk.Frame(main_container, bg=self.colors['card'], width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Configurations table in right frame
        table_title = tk.Label(right_frame, text="Configurations to Add:", 
                              font=('Segoe UI', 12, 'bold'), fg=self.colors['text'], bg=self.colors['card'])
        table_title.pack(anchor='w', pady=(0, 10))
        
        # Table container
        table_container = tk.Frame(right_frame, bg=self.colors['card'])
        table_container.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create table
        columns = ['ID', 'Architecture', 'Layers']
        self.config_tree = ttk.Treeview(table_container, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.config_tree.heading(col, text=col)
            if col == 'ID':
                self.config_tree.column(col, width=50)
            elif col == 'Architecture':
                self.config_tree.column(col, width=150)
            else:
                self.config_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.config_tree.yview)
        self.config_tree.configure(yscrollcommand=scrollbar.set)
        
        self.config_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons in right frame with modern styling
        button_frame = tk.Frame(right_frame, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        clear_button = tk.Button(button_frame, text="Clear All", 
                                command=self.clear_configurations,
                                font=('Segoe UI', 11), bg='#dc3545', fg='white', padx=20, pady=8)
        clear_button.pack(fill=tk.X, pady=(0, 10))
        
        cancel_button = tk.Button(button_frame, text="Cancel", 
                                 command=self.window.destroy,
                                 font=('Segoe UI', 11), bg='#6c757d', fg='white', padx=20, pady=8)
        cancel_button.pack(fill=tk.X, pady=(0, 10))
        
        ok_button = tk.Button(button_frame, text="Add to Sweep", 
                             command=self.confirm_add,
                             font=('Segoe UI', 11, 'bold'), bg=self.colors['accent'], fg='white', padx=20, pady=8)
        ok_button.pack(fill=tk.X)
    
    def setup_single_tab(self, parent_frame):
        """Setup single configuration tab with modern styling"""
        # Input frame
        input_frame = tk.Frame(parent_frame, bg=self.colors['card'])
        input_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Number of layers
        tk.Label(input_frame, text="Number of Hidden Layers:", 
                font=('Segoe UI', 12), fg=self.colors['text'], bg=self.colors['card']).grid(row=0, column=0, sticky='w', pady=5)
        
        self.layers_var = tk.StringVar(value="2")
        self.layers_spinbox = tk.Spinbox(input_frame, from_=1, to=10, textvariable=self.layers_var,
                                        font=('Segoe UI', 11), width=10, command=self.update_layer_inputs)
        self.layers_spinbox.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # Dynamic layer inputs frame
        self.layers_frame = tk.Frame(input_frame, bg=self.colors['card'])
        self.layers_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=10)
        
        self.layer_entries = []
        self.update_layer_inputs()
        
        # Add button
        add_button = tk.Button(input_frame, text="Add Configuration", 
                              command=self.add_configuration,
                              font=('Segoe UI', 11, 'bold'),
                              bg=self.colors['primary'], fg='white', padx=20, pady=5)
        add_button.grid(row=2, column=0, columnspan=2, pady=10)
    
    def setup_bulk_tab(self, parent_frame):
        """Setup bulk configuration tab with modern styling"""
        # Instructions
        instructions_frame = tk.Frame(parent_frame, bg=self.colors['card'])
        instructions_frame.pack(fill=tk.X, padx=20, pady=10)
        
        instruction_text = """Bulk Configuration Instructions:
Enter each layer's neuron options separated by commas, one layer per line.

Example for 2-layer architectures:
5, 10, 15, 20
8, 16, 32

This will create all combinations: 5x8, 5x16, 5x32, 10x8, 10x16, etc.

Example for 3-layer architectures:
10, 20
5, 10, 15  
4, 8

This creates: 10x5x4, 10x5x8, 10x10x4, 10x10x8, etc."""

        instruction_label = tk.Label(instructions_frame, text=instruction_text, 
                                    font=('Segoe UI', 10), fg=self.colors['secondary_text'], bg=self.colors['card'],
                                    justify=tk.LEFT, wraplength=800)
        instruction_label.pack(anchor='w')
        
        # Text area for bulk input
        text_frame = tk.Frame(parent_frame, bg=self.colors['card'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(text_frame, text="Enter layer configurations:", 
                font=('Segoe UI', 12, 'bold'), fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w')
        
        # Text widget with scrollbar
        text_container = tk.Frame(text_frame, bg=self.colors['card'])
        text_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.bulk_text = tk.Text(text_container, height=8, font=('Consolas', 11),
                                bg='white', fg=self.colors['text'], insertbackground=self.colors['primary'])
        bulk_scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=self.bulk_text.yview)
        self.bulk_text.configure(yscrollcommand=bulk_scrollbar.set)
        
        self.bulk_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bulk_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Example button and process button
        button_frame = tk.Frame(text_frame, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=5)
        
        example_button = tk.Button(button_frame, text="Load Example", 
                                  command=self.load_example,
                                  font=('Segoe UI', 10), bg='#6c757d', fg='white', padx=15)
        example_button.pack(side=tk.LEFT)
        
        process_button = tk.Button(button_frame, text="Process Bulk Input", 
                                  command=self.process_bulk_input,
                                  font=('Segoe UI', 11, 'bold'),
                                  bg=self.colors['primary'], fg='white', padx=20, pady=5)
        process_button.pack(side=tk.RIGHT)
    
    def update_layer_inputs(self):
        """Update layer input fields based on number of layers"""
        # Clear existing
        for widget in self.layers_frame.winfo_children():
            widget.destroy()
        
        self.layer_entries = []
        num_layers = int(self.layers_var.get())
        
        for i in range(num_layers):
            tk.Label(self.layers_frame, text=f"Layer {i+1} Neurons:", 
                    font=('Segoe UI', 11), fg=self.colors['text'], bg=self.colors['card']).grid(row=i, column=0, sticky='w', pady=2)
            
            entry = tk.Entry(self.layers_frame, font=('Segoe UI', 11), width=10)
            entry.grid(row=i, column=1, padx=(10, 0), pady=2)
            self.layer_entries.append(entry)
    
    def add_configuration(self):
        """Add current configuration to the list"""
        try:
            num_layers = int(self.layers_var.get())
            neurons = []
            
            for entry in self.layer_entries:
                value = entry.get().strip()
                if not value:
                    raise ValueError("All layer fields must be filled")
                neurons.append(int(value))
            
            # Create configuration in the format expected [n1, n2, n3, ..., 0, 0, total_layers]
            max_layers = 10  # Support up to 10 layers
            config = neurons + [0] * (max_layers - len(neurons)) + [num_layers]
            
            # Check for duplicates
            if config in self.configurations:
                tk.messagebox.showwarning("Duplicate", "This configuration already exists in the list")
                return
            
            self.configurations.append(config)
            
            # Add to table
            config_id = len(self.configurations)
            arch_label = "x".join(map(str, neurons))
            
            self.config_tree.insert('', 'end', values=(config_id, arch_label, num_layers))
            
            # Clear entries
            for entry in self.layer_entries:
                entry.delete(0, tk.END)
            
        except ValueError as e:
            tk.messagebox.showerror("Error", f"Invalid input: {str(e)}")
    
    def clear_configurations(self):
        """Clear all configurations"""
        self.configurations = []
        for item in self.config_tree.get_children():
            self.config_tree.delete(item)
    
    def confirm_add(self):
        """Confirm and add configurations to main sweep"""
        if not self.configurations:
            tk.messagebox.showwarning("No Configurations", "Please add at least one configuration")
            return
        
        self.parent.add_new_configurations(self.configurations)
        tk.messagebox.showinfo("Success", f"Added {len(self.configurations)} configurations to the sweep")
        self.window.destroy()
    
    def load_example(self):
        """Load example text in bulk input"""
        example_text = """5, 10, 15, 20
8, 16, 32"""
        self.bulk_text.delete(1.0, tk.END)
        self.bulk_text.insert(1.0, example_text)
    
    def process_bulk_input(self):
        """Process bulk input and generate all combinations"""
        try:
            text_content = self.bulk_text.get(1.0, tk.END).strip()
            if not text_content:
                tk.messagebox.showwarning("Empty Input", "Please enter layer configurations")
                return
            
            # Parse input
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            layer_options = []
            
            for line in lines:
                # Parse comma-separated values
                values = [int(x.strip()) for x in line.split(',') if x.strip()]
                if not values:
                    continue
                layer_options.append(values)
            
            if not layer_options:
                tk.messagebox.showwarning("Invalid Input", "No valid layer configurations found")
                return
            
            # Generate all combinations
            from itertools import product
            combinations = list(product(*layer_options))
            
            if not combinations:
                tk.messagebox.showwarning("No Combinations", "No combinations could be generated")
                return
            
            # Convert to expected format and add to configurations
            added_count = 0
            max_layers = 10  # Support up to 10 layers
            
            for combo in combinations:
                num_layers = len(combo)
                neurons = list(combo)
                
                # Create configuration in expected format [n1, n2, n3, ..., 0, 0, total_layers]
                config = neurons + [0] * (max_layers - len(neurons)) + [num_layers]
                
                # Check for duplicates
                if config not in self.configurations:
                    self.configurations.append(config)
                    added_count += 1
                    
                    # Add to table
                    config_id = len(self.configurations)
                    arch_label = "x".join(map(str, neurons))
                    
                    self.config_tree.insert('', 'end', values=(config_id, arch_label, num_layers))
            
            if added_count > 0:
                tk.messagebox.showinfo("Success", f"Added {added_count} configurations from bulk input\n"
                                                f"Generated from {len(combinations)} combinations\n"
                                                f"Skipped {len(combinations) - added_count} duplicates")
            else:
                tk.messagebox.showwarning("No New Configurations", "All generated configurations were duplicates")
            
            # Clear the text area
            self.bulk_text.delete(1.0, tk.END)
            
        except ValueError as e:
            tk.messagebox.showerror("Parse Error", f"Invalid input format: {str(e)}\n\n"
                                                   "Please use numbers separated by commas, one layer per line")
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")

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