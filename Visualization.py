import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from FullFwdPropagation import full_forward_propagation
from itertools import islice
import os
import pickle

# Set professional plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
        
sns.set_palette("husl")

def visualize_NN_results(pkl_file=None, output_dir="visualizations"):
    """
    Create comprehensive visualization of neural network training results from PKL file.
    
    Parameters:
    -----------
    pkl_file : str, optional
        Path to PKL file. If None, looks for most recent PKL file in current directory
    output_dir : str
        Directory to save plots
    """
    
    # Find PKL file if not specified
    if pkl_file is None:
        # First look for files with the new naming convention
        trained_pkl_files = [f for f in os.listdir('.') if f.startswith('Trained_DNN_') and f.endswith('.pkl')]
        
        if trained_pkl_files:
            # Get most recent Trained_DNN_ file
            pkl_file = max(trained_pkl_files, key=os.path.getmtime)
            print(f"Using most recent trained model: {pkl_file}")
        else:
            # Fallback to any PKL file
            pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
            if not pkl_files:
                print("ERROR: No PKL files found! Train a model first.")
                return False
            
            # Get most recent PKL file
            pkl_file = max(pkl_files, key=os.path.getmtime)
            print(f"Using most recent PKL file: {pkl_file}")
    
    if not os.path.exists(pkl_file):
        print(f"ERROR: PKL file {pkl_file} not found!")
        return False
    
    print(f"Loading training data from {pkl_file}...")
    
    # Load PKL data
    with open(pkl_file, 'rb') as f:
        stored_data = pickle.load(f)
    
    # Extract all necessary data
    X_train = stored_data['X']
    Y_train = stored_data['Y']
    X_valid = stored_data['X_valid']
    Y_valid = stored_data['Y_valid']
    params_values = stored_data['params_values']
    nn = stored_data['nn']
    min_data = stored_data['min_data']
    max_data = stored_data['max_data']
    outputIndexEntry = stored_data['outputIndexEntry']
    loss_history = stored_data['loss_history']
    headers = stored_data['headers']
    
    # Calculate R² scores for validation
    network_layers = dict(islice(nn.items(), 1, None))
    Y_hat_train, _ = full_forward_propagation(X_train, params_values, network_layers)
    Y_hat_valid, _ = full_forward_propagation(X_valid, params_values, network_layers)
    
    # Calculate R² per variable
    from Train import train_fidelity_r2
    accuracy_per_variable = []
    for i in range(Y_valid.shape[1]):
        r2 = train_fidelity_r2(Y_valid[:, i], Y_hat_valid[:, i])
        accuracy_per_variable.append(r2 / 100.0)  # Convert to 0-1 scale
    
    # Extract output variable names
    output_names = [headers[i] if i < len(headers) else f"Output_{i}" for i in outputIndexEntry]
    
    print(f"Data loaded successfully!")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Validation samples: {X_valid.shape[0]:,}")
    print(f"   Input features: {X_train.shape[1]}")
    print(f"   Output variables: {Y_train.shape[1]}")
    print(f"   Training epochs: {len(loss_history):,}")
    print(f"   Average R² score: {np.mean(accuracy_per_variable):.3f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visualizations
    print(f"\nGenerating comprehensive visualizations...")
    
    create_architecture_diagram(nn, output_dir)
    create_training_analysis(loss_history, output_dir)
    create_prediction_analysis(Y_train, Y_hat_train, Y_valid, Y_hat_valid, 
                             output_names, accuracy_per_variable, output_dir)
    create_error_analysis(Y_train, Y_hat_train, Y_valid, Y_hat_valid, 
                         output_names, output_dir)
    create_performance_summary(accuracy_per_variable, output_names, 
                             loss_history, nn, output_dir)
    
    print(f"\nVisualization complete! All plots saved in '{output_dir}/' directory")
    print(f"Generated files:")
    print(f"   - network_architecture.png")
    print(f"   - training_analysis.png") 
    print(f"   - prediction_analysis.png")
    print(f"   - error_analysis.png")
    print(f"   - performance_summary.png")
    
    return True


def create_architecture_diagram(nn, output_dir):
    """Create neural network architecture visualization."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract layer information
    layers = []
    layer_names = []
    
    # Get input dimension
    input_dim = nn["NeuralNetworkModel"]["inputEntryIndices"]
    layers.append(len(input_dim))
    layer_names.append("Input")
    
    # Get hidden layers
    network_layers = dict(islice(nn.items(), 1, None))
    for layer_name, layer_info in network_layers.items():
        if "Layer" in layer_name and layer_name != "OutputLayer":
            layers.append(layer_info["output_dim"])
            layer_names.append(layer_name.replace("Layer", ""))
    
    # Get output layer
    output_dim = nn["NeuralNetworkModel"]["outputEntryIndices"] 
    layers.append(len(output_dim))
    layer_names.append("Output")
    
    # Calculate positions
    n_layers = len(layers)
    max_neurons = max(layers)
    
    # Plot network architecture
    layer_colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    
    for i, (n_neurons, color) in enumerate(zip(layers, layer_colors)):
        x = i * 2.5
        
        # Draw main neurons
        y_positions = np.linspace(-max_neurons/2, max_neurons/2, n_neurons)
        
        for j, y in enumerate(y_positions):
            circle = plt.Circle((x, y), 0.3, color=color, alpha=0.7)
            ax.add_patch(circle)
        
        # Add bias neuron for hidden layers (not input or output)
        if i > 0 and i < n_layers - 1:
            bias_y = max_neurons/2 + 1.5
            bias_circle = plt.Circle((x, bias_y), 0.2, color='orange', alpha=0.8)
            ax.add_patch(bias_circle)
            ax.text(x, bias_y, 'b', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
        # Draw connections to next layer
        if i < n_layers - 1:
            next_neurons = layers[i + 1]
            next_y_positions = np.linspace(-max_neurons/2, max_neurons/2, next_neurons)
            
            # Connections from main neurons
            for y1 in y_positions:
                for y2 in next_y_positions:
                    ax.plot([x + 0.3, x + 2.5 - 0.3], [y1, y2], 
                           'k-', alpha=0.1, linewidth=0.5)
            
            # Connections from bias neuron (if exists)
            if i > 0 and i < n_layers - 1:
                bias_y = max_neurons/2 + 1.5
                for y2 in next_y_positions:
                    ax.plot([x + 0.2, x + 2.5 - 0.3], [bias_y, y2], 
                           'orange', alpha=0.3, linewidth=0.3)
        
        # Add layer labels
        ax.text(x, -max_neurons/2 - 1, f"{layer_names[i]}\n({n_neurons})", 
               ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Format plot
    ax.set_xlim(-1, (n_layers-1) * 2.5 + 1)
    ax.set_ylim(-max_neurons/2 - 3, max_neurons/2 + 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Add architecture summary in bottom right
    total_params = sum(layer_info.get("output_dim", 0) * (prev_dim + 1) # +1 for bias
                      for i, (layer_name, layer_info) in enumerate(network_layers.items())
                      for prev_dim in [layers[i]] if "Layer" in layer_name)
    
    arch_text = f"Architecture: {' → '.join(map(str, layers))}\n"
    arch_text += f"Parameters: ~{total_params:,}\n"
    arch_text += f"Activations: {', '.join(set(layer['activation'] for layer in network_layers.values()))}\n"
    arch_text += f"Orange circles: Bias neurons"
    
    ax.text(0.98, 0.02, arch_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/network_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_training_analysis(loss_history, output_dir):
    """Create training loss analysis - log scale only."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(loss_history) + 1)
    
    # Loss curve in log scale only
    ax.semilogy(epochs, loss_history, color='#2E86C1', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Professional styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add key statistics as text annotation
    final_loss = loss_history[-1]
    initial_loss = loss_history[0]
    min_loss = min(loss_history)
    min_epoch = np.argmin(loss_history) + 1
    
    stats_text = f"Initial: {initial_loss:.2e}\n"
    stats_text += f"Final: {final_loss:.2e}\n"
    stats_text += f"Best: {min_loss:.2e} (Epoch {min_epoch})\n"
    stats_text += f"Improvement: {((initial_loss - final_loss) / initial_loss * 100):.1f}%"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_analysis.png", dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()


def create_prediction_analysis(Y_train, Y_hat_train, Y_valid, Y_hat_valid, 
                             output_names, accuracy_per_variable, output_dir):
    """Create prediction vs actual value analysis with Q1 publication quality."""
    
    # High-quality color palette for publications (colorblind-friendly)
    colors = {
        'training': '#1f77b4',      # Professional blue
        'validation': '#ff7f0e',    # Professional orange  
        'perfect': '#2c2c2c',       # Dark gray
        'error_band': '#d62728'     # Professional red
    }
    
    n_outputs = len(output_names)
    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_outputs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (output_name, r2_score) in enumerate(zip(output_names, accuracy_per_variable)):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Plot training data with high-quality styling
        ax.scatter(Y_train[:, i], Y_hat_train[:, i], alpha=0.7, s=15, 
                  label='Training', color=colors['training'], edgecolors='none')
        
        # Plot validation data
        ax.scatter(Y_valid[:, i], Y_hat_valid[:, i], alpha=0.7, s=15, 
                  label='Validation', color=colors['validation'], edgecolors='none')
        
        # Perfect prediction line
        ax.plot([0, 1], [0, 1], color=colors['perfect'], linestyle='--', 
               linewidth=1.5, alpha=0.8, label='Perfect Prediction')
        
        # ±5% error bands
        x_band = np.linspace(0, 1, 100)
        ax.fill_between(x_band, x_band - 0.05, x_band + 0.05, 
                       alpha=0.15, color=colors['error_band'], label='±5% Error')
        
        # Set limits to [0, 1] for normalized data
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        ax.set_xlabel(f'Actual {output_name}', fontsize=10)
        ax.set_ylabel(f'Predicted {output_name}', fontsize=10)
        ax.set_title(f'{output_name} (R² = {r2_score:.3f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set equal aspect ratio and professional styling
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Add professional spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    # Hide empty subplots
    for i in range(n_outputs, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    # No main title for publication quality
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{output_dir}/prediction_analysis.png", dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()


def create_error_analysis(Y_train, Y_hat_train, Y_valid, Y_hat_valid, output_names, output_dir):
    """Create error distribution and residual analysis."""
    
    n_outputs = len(output_names)
    fig = plt.figure(figsize=(12, 4*n_outputs))
    gs = GridSpec(n_outputs, 3, figure=fig)
    
    for i, output_name in enumerate(output_names):
        # Calculate residuals (difference between predicted and actual)
        train_residuals = Y_hat_train[:, i] - Y_train[:, i]
        valid_residuals = Y_hat_valid[:, i] - Y_valid[:, i]
        
        # Calculate percentage errors (avoid division by zero)
        train_pct_errors = 100 * train_residuals / np.where(np.abs(Y_train[:, i]) > 1e-8, Y_train[:, i], 1e-8)
        valid_pct_errors = 100 * valid_residuals / np.where(np.abs(Y_valid[:, i]) > 1e-8, Y_valid[:, i], 1e-8)
        
        # Remove extreme outliers for better visualization
        train_pct_errors = np.clip(train_pct_errors, -200, 200)
        valid_pct_errors = np.clip(valid_pct_errors, -200, 200)
        
        # 1. Residuals vs Actual (Residuals = difference between predicted and actual values)
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.scatter(Y_train[:, i], train_residuals, alpha=0.6, s=15, label='Training', color='blue')
        ax1.scatter(Y_valid[:, i], valid_residuals, alpha=0.6, s=15, label='Validation', color='red')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.8)
        ax1.set_xlabel(f'Actual {output_name}')
        ax1.set_ylabel('Residuals (Predicted - Actual)')
        ax1.set_title(f'Residuals vs Actual Values')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Residual distribution (should be centered around 0 for good model)
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.hist(train_residuals, bins=30, alpha=0.7, label='Training', color='blue', density=True)
        ax2.hist(valid_residuals, bins=30, alpha=0.7, label='Validation', color='red', density=True)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Residuals (Predicted - Actual)')
        ax2.set_ylabel('Density')
        ax2.set_title('Residual Distribution')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Percentage errors distribution
        ax3 = fig.add_subplot(gs[i, 2])
        if len(train_pct_errors) > 0 and len(valid_pct_errors) > 0:
            ax3.hist(train_pct_errors, bins=30, alpha=0.7, label='Training', color='blue', density=True)
            ax3.hist(valid_pct_errors, bins=30, alpha=0.7, label='Validation', color='red', density=True)
            ax3.axvline(x=0, color='k', linestyle='--', alpha=0.8)
            ax3.set_xlabel('Relative Errors (%)')
            ax3.set_ylabel('Density')
            ax3.set_title('Relative Error Distribution')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No valid percentage\nerrors to display', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Relative Error Distribution')
    
    plt.suptitle('Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_summary(accuracy_per_variable, output_names, loss_history, nn, output_dir):
    """Create overall performance summary dashboard."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. R² Scores Bar Chart
    ax1 = fig.add_subplot(gs[0, :2])
    colors = plt.cm.viridis(np.linspace(0, 1, len(output_names)))
    bars = ax1.bar(output_names, accuracy_per_variable, color=colors, alpha=0.8)
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance by Output Variable', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(1.0, max(accuracy_per_variable) * 1.1))
    
    # Add value labels on bars
    for bar, score in zip(bars, accuracy_per_variable):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Performance Classification
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Classify performance
    excellent = sum(1 for r2 in accuracy_per_variable if r2 >= 0.95)
    good = sum(1 for r2 in accuracy_per_variable if 0.85 <= r2 < 0.95)
    fair = sum(1 for r2 in accuracy_per_variable if 0.70 <= r2 < 0.85)
    poor = sum(1 for r2 in accuracy_per_variable if r2 < 0.70)
    
    performance_counts = [excellent, good, fair, poor]
    performance_labels = ['Excellent\n(R²≥0.95)', 'Good\n(0.85≤R²<0.95)', 
                         'Fair\n(0.70≤R²<0.85)', 'Poor\n(R²<0.70)']
    performance_colors = ['green', 'lightgreen', 'orange', 'red']
    
    # Filter out zero counts
    non_zero_counts = [(count, label, color) for count, label, color in 
                      zip(performance_counts, performance_labels, performance_colors) if count > 0]
    
    if non_zero_counts:
        counts, labels, colors = zip(*non_zero_counts)
        wedges, texts, autotexts = ax2.pie(counts, labels=labels, colors=colors, autopct='%1.0f',
                                          startangle=90, textprops={'fontsize': 8})
    
    ax2.set_title('Performance Classification', fontweight='bold')
    
    # 3. Training Summary
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Model summary
    avg_r2 = np.mean(accuracy_per_variable)
    best_r2 = max(accuracy_per_variable)
    worst_r2 = min(accuracy_per_variable)
    best_var = output_names[np.argmax(accuracy_per_variable)]
    worst_var = output_names[np.argmin(accuracy_per_variable)]
    
    summary_text = f"MODEL PERFORMANCE SUMMARY\n"
    summary_text += f"{'='*50}\n\n"
    summary_text += f"Overall Performance:\n"
    summary_text += f"  • Average R² Score: {avg_r2:.3f}\n"
    summary_text += f"  • Best Performance: {best_r2:.3f} ({best_var})\n"
    summary_text += f"  • Worst Performance: {worst_r2:.3f} ({worst_var})\n"
    summary_text += f"  • Performance Range: {best_r2 - worst_r2:.3f}\n\n"
    
    summary_text += f"Training Details:\n"
    summary_text += f"  • Total Epochs: {len(loss_history):,}\n"
    if loss_history:
        summary_text += f"  • Final Loss: {loss_history[-1]:.2e}\n"
        summary_text += f"  • Loss Reduction: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.1f}%\n"
    summary_text += f"  • Optimizer: {nn['NeuralNetworkModel'].get('UpdateMethod', 'Adam')}\n"
    summary_text += f"  • Learning Rate: {nn['NeuralNetworkModel'].get('learning_rate', 'N/A')}\n\n"
    
    # Performance recommendations
    if avg_r2 >= 0.90:
        recommendation = "EXCELLENT: Model shows excellent predictive performance across all outputs."
    elif avg_r2 >= 0.80:
        recommendation = "GOOD: Model shows good performance. Consider fine-tuning for better results."
    elif avg_r2 >= 0.70:
        recommendation = "FAIR: Model shows acceptable performance. Architecture optimization recommended."
    else:
        recommendation = "POOR: Model performance is below acceptable threshold. Redesign needed."
    
    summary_text += f"Recommendation: {recommendation}"
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    # 4. Individual Variable Performance
    ax4 = fig.add_subplot(gs[2, :])
    
    # Create detailed performance table
    var_data = []
    for i, (name, r2) in enumerate(zip(output_names, accuracy_per_variable)):
        if r2 >= 0.95:
            status = "Excellent"
        elif r2 >= 0.85:
            status = "Good"  
        elif r2 >= 0.70:
            status = "Fair"
        else:
            status = "Poor"
        var_data.append([name, f"{r2:.4f}", status])
    
    # Create table
    table = ax4.table(cellText=var_data,
                     colLabels=['Output Variable', 'R² Score', 'Performance'],
                     cellLoc='center',
                     loc='upper center',
                     colWidths=[0.3, 0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(output_names) + 1):
        for j in range(3):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 1:  # R² score column
                    r2_val = accuracy_per_variable[i-1]
                    if r2_val >= 0.95:
                        table[(i, j)].set_facecolor('#E8F5E8')
                    elif r2_val >= 0.85:
                        table[(i, j)].set_facecolor('#FFF8DC')
                    elif r2_val >= 0.70:
                        table[(i, j)].set_facecolor('#FFE4B5')
                    else:
                        table[(i, j)].set_facecolor('#FFE4E1')
    
    ax4.axis('off')
    ax4.set_title('Detailed Performance by Variable', fontweight='bold', pad=5, y=0.9)
    
    plt.suptitle('Neural Network Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    
    print("Neural Network Visualization Tool")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        # PKL file provided as argument
        pkl_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"
        visualize_NN_results(pkl_file, output_dir)
    else:
        # Use most recent PKL file
        visualize_NN_results()