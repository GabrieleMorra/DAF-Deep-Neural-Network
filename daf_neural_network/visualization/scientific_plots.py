"""
High-Quality Scientific Visualization Module for Deep Neural Networks
Designed for Q1 Journal Publication Standards

Author: Scientific Computing Module
Purpose: Generate publication-ready plots for DNN regression analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import pickle
import os
from itertools import islice
from scipy.stats import pearsonr, spearmanr
from scipy import signal
from sklearn.preprocessing import StandardScaler
import shap
# Import necessary modules for neural network operations
from ..core.forward_propagation import full_forward_propagation
from ..models.onnx_inference import infere, read, check

def visualize_NN_results(pkl_file=None, output_dir="visualizations", latex_var_names=None):
    """
    Generate high-quality scientific visualizations for neural network training results.
    Creates only two essential plots suitable for Q1 journal publications:
    1. Training convergence analysis (error history)
    2. SHAP analysis for model interpretability
    
    Parameters:
    -----------
    pkl_file : str, optional
        Path to PKL file containing training results
    output_dir : str
        Directory to save visualizations and data files
    latex_var_names : dict, optional
        Dictionary mapping variable names to LaTeX formatted names
        Example: {'Mach': r'$M_{\infty}$', 'Alpha': r'$\alpha$', 'Cl': r'$C_L$'}
    """
    
    # Find and load PKL file for basic data
    if pkl_file is None:
        trained_pkl_files = [f for f in os.listdir('.') if f.startswith('Trained_DNN_') and f.endswith('.pkl')]
        if trained_pkl_files:
            pkl_file = max(trained_pkl_files, key=os.path.getmtime)
        else:
            pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
            if not pkl_files:
                print("ERROR: No PKL files found! Train a model first.")
                return False
            pkl_file = max(pkl_files, key=os.path.getmtime)
    
    if not os.path.exists(pkl_file):
        print(f"ERROR: PKL file {pkl_file} not found!")
        return False
    
    print(f"Loading training data from: {pkl_file}")
    
    # Load training data from PKL (but not the predictions)
    with open(pkl_file, 'rb') as f:
        stored_data = pickle.load(f)
    
    # Extract metadata from PKL
    loss_history = stored_data['loss_history']
    headers = stored_data['headers']
    outputIndexEntry = stored_data['outputIndexEntry']
    nn = stored_data['nn']
    min_data = stored_data.get('min_data', None)
    max_data = stored_data.get('max_data', None)
    
    # Load ORIGINAL DIMENSIONAL DATA from CSV file specified in JSON
    input_filename = nn["NeuralNetworkModel"]["InputFileName"]
    delimiter = nn["NeuralNetworkModel"].get("Delimiter", ",")
    input_indices = nn["NeuralNetworkModel"]["inputEntryIndices"]
    output_indices = nn["NeuralNetworkModel"]["outputEntryIndices"]
    
    if not os.path.exists(input_filename):
        print(f"ERROR: Input data file {input_filename} not found!")
        return False
    
    # Load original CSV data
    import pandas as pd
    df = pd.read_csv(input_filename, delimiter=delimiter)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")
    
    # Extract original dimensional input and output data
    X_original = df.iloc[:, input_indices].values
    Y_original = df.iloc[:, output_indices].values
    
    # Use the same train/test split ratio as in training
    training_ratio = nn["NeuralNetworkModel"]["training_testing_ratio"]
    split_idx = int(len(X_original) * training_ratio)
    
    X_train = X_original[:split_idx]
    Y_train = Y_original[:split_idx]
    X_valid = X_original[split_idx:]
    Y_valid = Y_original[split_idx:]
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\nSuccessfully loaded original dimensional data:\n")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Validation samples: {X_valid.shape[0]:,}")
    print(f"   Input features: {X_train.shape[1]}")
    print(f"   Output variables: {Y_train.shape[1]}")
    
    # Find corresponding ONNX file
    pkl_basename = os.path.splitext(pkl_file)[0]
    onnx_file = f"{pkl_basename}.onnx"
    
    if not os.path.exists(onnx_file):
        print(f"ERROR: ONNX file {onnx_file} not found!")
        print("Available files:", [f for f in os.listdir('.') if f.endswith('.onnx')])
        return False
    
    print(f"Loading ONNX model from: {onnx_file}")
    
    # Load ONNX model and calculate predictions using existing function
    # ONNX model handles normalization internally (input: real data -> output: real data)
    onnx_session = None
    try:
        check(onnx_file, verbose=False)
        model, onnx_session = read(onnx_file)
        
        print(f"Successfully loaded ONNX model (handles normalization internally)")
        
        # Calculate predictions using existing infere function with original dimensional data
        Y_hat_train = []
        Y_hat_valid = []
        
        print(f"Computing training predictions with ONNX...")
        for i, x_sample in enumerate(X_train):
            prediction = infere(x_sample, onnx_session)
            Y_hat_train.append(prediction)
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{len(X_train)} training samples")
        
        print(f"Computing validation predictions with ONNX...")
        for i, x_sample in enumerate(X_valid):
            prediction = infere(x_sample, onnx_session)
            Y_hat_valid.append(prediction)
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{len(X_valid)} validation samples")
        
        Y_hat_train = np.array(Y_hat_train)
        Y_hat_valid = np.array(Y_hat_valid)
        
        print(f"Successfully calculated ONNX predictions (dimensional values)")
        print(f"   Training predictions shape: {Y_hat_train.shape}")
        print(f"   Validation predictions shape: {Y_hat_valid.shape}")
        
    except Exception as e:
        print(f"ERROR: Failed to load ONNX model: {e}")
        print("Cannot compute SHAP analysis without proper model")
        return False
    
    if onnx_session is None:
        print("ERROR: ONNX session not initialized properly")
        return False
    
    # Get variable names
    input_indices = nn["NeuralNetworkModel"]["inputEntryIndices"]
    input_names = [headers[i] for i in input_indices]
    output_names = [headers[i] for i in outputIndexEntry]
    
    print(f"\nOriginal variable names:")
    print(f"Input variables: {input_names}")
    print(f"Output variables: {output_names}")
    
    # Helper function to clean LaTeX names for display when LaTeX is disabled
    def clean_latex_name(name):
        """Remove LaTeX dollar signs and basic formatting when LaTeX is disabled"""
        if isinstance(name, str):
            # Remove dollar signs
            cleaned = name.replace('$', '')
            # Basic replacements for common LaTeX constructs (use ASCII)
            cleaned = cleaned.replace('\\alpha', 'alpha')
            cleaned = cleaned.replace('\\Delta', 'Delta')
            cleaned = cleaned.replace('\\text{', '').replace('}', '')
            cleaned = cleaned.replace('_{', '_').replace('^{', '^')
            # Remove remaining braces
            cleaned = cleaned.replace('{', '').replace('}', '')
            # If result is empty after cleaning, return the cleaned version as is
            return cleaned if cleaned.strip() else name.replace('$', '')
        return name
    
    # Apply LaTeX formatting if provided
    if latex_var_names:
        if isinstance(latex_var_names, dict):
            # Dictionary mapping: original_name -> latex_name
            input_names_display = [latex_var_names.get(name, name) for name in input_names]
            output_names_display = [latex_var_names.get(name, name) for name in output_names]
        elif isinstance(latex_var_names, list):
            # List format: check if it matches inputs only or all variables
            all_names = input_names + output_names
            if len(latex_var_names) == len(input_names):
                # List contains only input names
                input_names_display = latex_var_names
                output_names_display = output_names
                print(f"Using LaTeX names for input variables only")
            elif len(latex_var_names) == len(all_names):
                # List contains all variable names (inputs + outputs)
                input_names_display = latex_var_names[:len(input_names)]
                output_names_display = latex_var_names[len(input_names):]
                print(f"Using LaTeX names from list (assuming same order)")
            else:
                print(f"WARNING: LaTeX list length ({len(latex_var_names)}) doesn't match inputs ({len(input_names)}) or all variables ({len(all_names)})")
                input_names_display = input_names
                output_names_display = output_names
        else:
            print(f"WARNING: latex_var_names must be dict or list")
            input_names_display = input_names
            output_names_display = output_names
        
        # Clean LaTeX names for display only when LaTeX rendering is disabled
        # Since we're using LaTeX rendering, keep the original LaTeX syntax
        # input_names_display = [clean_latex_name(name) for name in input_names_display]
        # output_names_display = [clean_latex_name(name) for name in output_names_display]
    else:
        input_names_display = input_names
        output_names_display = output_names
    
    print(f"Display names used: {input_names_display + output_names_display}")
    
    # Show mapping for user verification
    if latex_var_names:
        print(f"\nVariable Name Mapping (Dataset -> Display):")
        print("-" * 50)
        for i, (orig, disp) in enumerate(zip(input_names, input_names_display)):
            print(f"  Input {i+1:2d}: {orig:15} -> {disp}")
        for i, (orig, disp) in enumerate(zip(output_names, output_names_display)):
            print(f"  Output{i+1:2d}: {orig:15} -> {disp}")
        print("-" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nDataset Information:")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Validation samples: {X_valid.shape[0]:,}")
    print(f"Input features: {len(input_names)}")
    print(f"Output variables: {len(output_names)}")
    print(f"Training epochs: {len(loss_history):,}")
    
    # Debug: Compare ONNX predictions with original dimensional data
    print(f"\nDimensional Data Analysis:")
    print(f"Input data ranges:")
    for i, name in enumerate(input_names):
        x_min, x_max = np.min(X_train[:, i]), np.max(X_train[:, i])
        x_std = np.std(X_train[:, i])
        print(f"  {name:20}: [{x_min:10.4f}, {x_max:10.4f}], std={x_std:10.4f}")
    
    print(f"\nOutput data ranges:")
    for i, name in enumerate(output_names):
        y_min, y_max = np.min(Y_train[:, i]), np.max(Y_train[:, i])
        y_std = np.std(Y_train[:, i])
        print(f"  {name:20}: [{y_min:10.4f}, {y_max:10.4f}], std={y_std:10.4f}")
    
    
    print(f"\nOutput comparison (Original vs ONNX predictions):")
    for i, name in enumerate(output_names):
        orig_min, orig_max = np.min(Y_train[:, i]), np.max(Y_train[:, i])
        pred_min, pred_max = np.min(Y_hat_train[:, i]), np.max(Y_hat_train[:, i])
        orig_std = np.std(Y_train[:, i])
        pred_std = np.std(Y_hat_train[:, i])
        
        print(f"  {name:20}:")
        print(f"    Original: [{orig_min:10.4f}, {orig_max:10.4f}], std={orig_std:10.4f}")
        print(f"    ONNX Pred:[{pred_min:10.4f}, {pred_max:10.4f}], std={pred_std:10.4f}")
        
        if pred_std < 1e-6:
            print(f"    WARNING: ONNX predictions are nearly constant!")
        else:
            # Calculate correlation between original and predicted
            correlation = np.corrcoef(Y_train[:, i], Y_hat_train[:, i])[0, 1]
            print(f"    Correlation: {correlation:.4f}")
    
    # Generate the two essential scientific plots
    print(f"\n")
    
    # 1. Training Error History Plot
    create_training_convergence_plot(loss_history, output_dir)
    
    # 2. SHAP Analysis Plot (using ONNX model predictions for explainability)
    create_shap_analysis_plot(X_train, Y_train, X_valid, Y_valid, 
                             input_names, output_names, output_dir, onnx_session,
                             input_names_display, output_names_display)
    
    print(f"\nVisualization complete! Files saved in '{output_dir}/' directory:")
    print(f"   - training_convergence.png (Training error history)")
    print(f"   - shap_analysis_[output].png (SHAP beeswarm plots)")
    print(f"   - shap_importance_[output].png (SHAP feature importance)")
    print(f"   - training_convergence_data.txt (Training data)")
    
    return True


def create_training_convergence_plot(loss_history, output_dir):
    """
    Create publication-quality training convergence plot showing error history.
    No textboxes - all statistics printed to console.
    """
    
    # Set scientific publication parameters with Computer Modern for math only
    plt.rcParams.update({
        'text.usetex': False,  # Disable full LaTeX to avoid tight_layout errors
        'font.family': 'serif',
        'mathtext.fontset': 'cm',  # Use Computer Modern for math text only
        'font.size': 18,
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 22,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    epochs = np.arange(1, len(loss_history) + 1)
    
    # Calculate and plot only smoothed trend line
    if len(loss_history) > 50:
        window_length = min(51, len(loss_history) // 15)
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= 3:
            smoothed = signal.savgol_filter(loss_history, window_length, 3)
        else:
            smoothed = loss_history
    else:
        smoothed = loss_history
    
    # Plot only smoothed version with bold black line
    ax.semilogy(epochs, smoothed, color='black', linewidth=3.0, alpha=0.9)
    
    # Scientific formatting
    ax.set_xlabel('Training Epoch', fontweight='bold')
    ax.set_ylabel('Loss Function Value', fontweight='bold')
    ax.set_title('Neural Network Training Convergence', fontweight='bold', pad=20)
    
    # Professional appearance
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    ax.tick_params(axis='both', which='major', width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1.0, length=3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_convergence.png", dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_dir}/training_convergence.pdf", 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Calculate and print comprehensive statistics to console
    final_loss = loss_history[-1]
    initial_loss = loss_history[0]
    min_loss = min(loss_history)
    min_epoch = np.argmin(loss_history) + 1
    
    # Convergence analysis
    last_10pct = max(1, int(len(loss_history) * 0.1))
    convergence_variance = np.var(loss_history[-last_10pct:])
    improvement_ratio = (initial_loss - final_loss) / initial_loss * 100
    loss_gradient = np.gradient(loss_history)
    avg_learning_rate = np.mean(np.abs(loss_gradient))
    
    print(f"\n" + "="*70)
    print(f"TRAINING CONVERGENCE ANALYSIS")
    print(f"="*70)
    print(f"Initial Loss:              {initial_loss:.8e}")
    print(f"Final Loss:                {final_loss:.8e}")
    print(f"Minimum Loss Achieved:     {min_loss:.8e}")
    print(f"Best Loss at Epoch:        {min_epoch:,}")
    print(f"Total Training Epochs:     {len(loss_history):,}")
    print(f"Loss Reduction:            {improvement_ratio:.3f}%")
    print(f"Convergence Variance:      {convergence_variance:.8e}")
    print(f"Average Learning Rate:     {avg_learning_rate:.8e}")
    print(f"Loss Stability (last 10%): {np.std(loss_history[-last_10pct:]):.8e}")
    
    # Convergence assessment
    is_converged = convergence_variance < final_loss * 0.001
    print(f"Convergence Status:        {'CONVERGED' if is_converged else 'STILL LEARNING'}")
    print(f"="*70)
    
    # Generate data file with training history
    with open(f"{output_dir}/training_convergence_data.txt", 'w') as f:
        f.write("Epoch\tLoss_Value\tSmoothed_Trend\n")
        
        # Calculate smoothed values for data file
        if len(loss_history) > 50:
            window_length = min(51, len(loss_history) // 15)
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= 3:
                smoothed = signal.savgol_filter(loss_history, window_length, 3)
            else:
                smoothed = loss_history
        else:
            smoothed = loss_history
            
        for i, (loss, smooth) in enumerate(zip(loss_history, smoothed), 1):
            f.write(f"{i}\t{loss:.10e}\t{smooth:.10e}\n")


def create_shap_analysis_plot(X_train, Y_train, X_valid, Y_valid, 
                             input_names, output_names, output_dir, onnx_session,
                             input_names_display, output_names_display):
    """
    Create SHAP (SHapley Additive exPlanations) analysis plots for neural network explanability.
    
    SHAP values explain the contribution of each feature to individual predictions,
    providing model-agnostic explanations for the neural network's decisions.
    
    Parameters:
    -----------
    onnx_session : ONNX runtime session for model predictions
    """
    
    print(f"\nGenerating SHAP explanations for model interpretability...")
    
    # Create cache filename based on ONNX model and data
    import hashlib
    import time
    
    # Create combined data for cache key generation
    X_combined_temp = np.vstack([X_train, X_valid])
    Y_combined_temp = np.vstack([Y_train, Y_valid])
    
    # Use ONNX file timestamp and combined data shape for cache key
    onnx_mtime = os.path.getmtime(os.path.join(os.getcwd(), 
        [f for f in os.listdir('.') if f.startswith('Trained_DNN_') and f.endswith('.onnx')][0]))
    cache_key = f"{X_combined_temp.shape}_{Y_combined_temp.shape}_{onnx_mtime}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_file = os.path.join(output_dir, f"shap_cache_{cache_hash}.pkl")
    
    # Try to load cached SHAP values
    shap_values = None
    X_sample = None
    
    if os.path.exists(cache_file):
        try:
            print(f"Loading cached SHAP analysis from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                shap_values = cached_data['shap_values']
                X_sample = cached_data['X_sample']
                n_cached_samples = len(X_sample) if X_sample is not None else 0
            print(f"\n{'='*60}")
            print(f"🔄 CACHE FOUND!")
            print(f"Using previously computed SHAP values for {n_cached_samples} samples.")
            print(f"This will skip the interactive sample selection process.")
            print(f"{'='*60}\n")
            input("⚡ Press ENTER to continue with cached data...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Computing fresh SHAP analysis...")
            shap_values = None
    
    # Compute SHAP values if not cached
    if shap_values is None:
        # Set publication parameters with Computer Modern for math only
        plt.rcParams.update({
            'text.usetex': False,  # Disable full LaTeX to avoid tight_layout errors
            'font.family': 'serif',
            'mathtext.fontset': 'cm',  # Use Computer Modern for math text only
            'font.size': 18,
            'axes.linewidth': 1.5,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 22,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white'
        })
        
        # Create wrapper function for ONNX model
        def onnx_predict(X):
            """Wrapper function for ONNX predictions compatible with SHAP"""
            predictions = []
            for x_sample in X:
                pred = infere(x_sample, onnx_session)
                predictions.append(pred)
            return np.array(predictions)
        
        # Combine train and test data for SHAP analysis
        X_combined = np.vstack([X_train, X_valid])
        Y_combined = np.vstack([Y_train, Y_valid])
        
        # Interactive sample size selection
        max_samples = len(X_combined)
        print(f"Total available samples (train + test): {max_samples:,}")
        
        try:
            user_input = input(f"Enter number of samples for SHAP analysis (0-{max_samples}): ")
            n_samples = int(float(user_input))  # Convert to float first to handle decimals, then to int
            
            # Validate and clamp the input
            if n_samples < 0:
                n_samples = 0
            elif n_samples > max_samples:
                n_samples = max_samples
                
            print(f"Using {n_samples} samples for SHAP analysis")
            
        except (ValueError, KeyboardInterrupt):
            # Default fallback if input is invalid or user cancels
            n_samples = min(500, max_samples)
            print(f"Invalid input or cancelled. Using default: {n_samples} samples")
        
        X_sample = X_combined[:n_samples]
        
        print(f"Computing SHAP values for {n_samples} samples (this may take a few minutes)...")
        start_time = time.time()
        
        # Initialize SHAP explainer with background data (use min of 100 or n_samples)
        n_background = min(100, n_samples)
        explainer = shap.Explainer(onnx_predict, X_sample[:n_background])
        
        # Calculate SHAP values for all selected samples (use min of 200 or n_samples)
        n_explain = min(200, n_samples)
        shap_values = explainer(X_sample[:n_explain])
        
        elapsed_time = time.time() - start_time
        print(f"SHAP analysis complete in {elapsed_time:.1f} seconds!")
        
        # Cache the results for future use
        try:
            cached_data = {
                'shap_values': shap_values,
                'X_sample': X_sample,
                'n_samples_used': n_samples,
                'computation_time': elapsed_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"SHAP results cached to {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    # Set publication parameters with Computer Modern for math only
    plt.rcParams.update({
        'text.usetex': False,  # Disable full LaTeX to avoid tight_layout errors
        'font.family': 'serif',
        'mathtext.fontset': 'cm',  # Use Computer Modern for math text only
        'font.size': 18,
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 22,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white'
    })
    
    # Create summary plots for each output
    for output_idx, output_name in enumerate(output_names):
        # Extract SHAP values for this output
        if len(shap_values.values.shape) == 3:
            shap_vals = shap_values.values[:, :, output_idx]
        else:
            shap_vals = shap_values.values
        
        # Create combined SHAP plot with analysis on left and importance on right
        # Use rectangular figure with wider proportions and custom subplot widths
        fig = plt.figure(figsize=(20, 8))
        
        # Create custom grid: left plot wider than right plot
        gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.2, 0.1])
        
        # Left plot: SHAP beeswarm analysis (takes 2.5 parts of width)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.sca(ax1)
        
        # Create SHAP beeswarm plot without automatic colorbar with larger font and medium dots
        n_plot_samples = min(len(X_sample), shap_vals.shape[0])
        shap.plots.beeswarm(shap.Explanation(
            values=shap_vals,
            data=X_sample[:n_plot_samples],
            feature_names=input_names_display,
        ), show=False, plot_size=(15, 8), color_bar=False, s=35)
        
        # Increase font sizes for the left plot
        ax1.tick_params(axis='both', which='major', labelsize=16)  # Increase tick label size
        ax1.tick_params(axis='x', which='major', labelsize=16)     # X-axis tick labels
        ax1.tick_params(axis='y', which='major', labelsize=20)     # Y-axis tick labels (LaTeX feature names) - increased further
        
        # Increase axis label sizes
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=18, fontweight='bold')
        if ax1.get_ylabel():
            ax1.set_ylabel(ax1.get_ylabel(), fontsize=18, fontweight='bold')
        
        # Change the vertical zero line from gray to black
        ax1.axvline(x=0, color='black', linewidth=1.5, alpha=0.8)
        
        # Get the y-axis limits and tick positions from the left plot for alignment
        left_ylim = ax1.get_ylim()
        left_yticks = ax1.get_yticks()
        
        # Force remove any colorbars that might have been created
        # Get all axes in the figure and remove narrow ones (likely colorbars)
        current_axes = fig.get_axes().copy()
        for ax in current_axes:
            if ax != ax1:  # Don't remove our main plot
                try:
                    pos = ax.get_position()
                    # Remove if it's narrow (likely a colorbar) or specifically labeled as colorbar
                    if pos.width < 0.1 or 'colorbar' in str(ax).lower():
                        ax.remove()
                except:
                    pass
        
        # Right plot: Feature importance with matching colors (takes 1.2 parts of width)
        ax2 = fig.add_subplot(gs[0, 1])
        plt.sca(ax2)
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.mean(np.abs(shap_vals), axis=0)
        
        # Sort features by importance (same order as beeswarm plot)
        sorted_indices = np.argsort(mean_shap)
        sorted_names_display = [input_names_display[i] for i in sorted_indices]
        sorted_importance = mean_shap[sorted_indices]
        
        # GROUPING MANAGEMENT: Check if SHAP grouped features in the left plot
        # Extract y-labels from left SHAP plot to see if there are groupings
        left_labels = [label.get_text() for label in ax1.get_yticklabels()]
        
        # Search for "Sum of X other features" pattern or similar
        grouped_features_count = 0
        grouped_pattern_found = False
        for label in left_labels:
            if "Sum of" in label and "other" in label:
                # Extract number of grouped features (e.g., "Sum of 4 other features")
                import re
                match = re.search(r'Sum of (\d+) other', label)
                if match:
                    grouped_features_count = int(match.group(1))
                    grouped_pattern_found = True
                    break
        
        # If SHAP grouped features, adapt the right plot accordingly
        if grouped_pattern_found and grouped_features_count > 0:
            print(f"[SHAP GROUPING] Detected {grouped_features_count} features grouped in left plot")
            
            # Group the last N features (those with lower importance)
            if len(sorted_importance) > grouped_features_count:
                # Separate individual features from features to be grouped
                individual_features = len(sorted_importance) - grouped_features_count
                
                # Individual features (most important ones)
                individual_importance = sorted_importance[grouped_features_count:]
                individual_names = sorted_names_display[grouped_features_count:]
                
                # Features to be grouped (least important ones)
                grouped_importance_values = sorted_importance[:grouped_features_count]
                grouped_names = sorted_names_display[:grouped_features_count]
                
                # Calculate mean importance of grouped features
                grouped_importance_mean = np.mean(grouped_importance_values)
                
                # Create new arrays for plotting
                final_importance = np.concatenate([[grouped_importance_mean], individual_importance])
                final_names = [f"Sum of {grouped_features_count} other features"] + individual_names
                
                print(f"[GROUPING] Individual features: {len(individual_names)}")
                print(f"[GROUPING] Grouped features: {grouped_features_count} -> mean importance: {grouped_importance_mean:.6f}")
            else:
                # Fallback if too few features
                final_importance = sorted_importance
                final_names = sorted_names_display
        else:
            # No grouping necessary
            final_importance = sorted_importance
            final_names = sorted_names_display
        
        # Create custom color gradient: max=#FF0052, center=#9B24AE, min=#008AFB
        from matplotlib.colors import LinearSegmentedColormap
        
        # Define custom colors
        colors_hex = ['#008AFB', '#9B24AE', '#FF0052']  # min -> center -> max
        colors_rgb = [tuple(int(c[1:][i:i+2], 16)/255.0 for i in (0, 2, 4)) for c in colors_hex]
        
        # Create custom colormap
        custom_cmap = LinearSegmentedColormap.from_list('custom', colors_rgb, N=256)
        
        # Apply colors based on normalized importance (use final_importance)
        normalized_importance = final_importance / final_importance.max()
        colors = [custom_cmap(val) for val in normalized_importance]
        
        # Create horizontal bar plot with grouped features if necessary
        bars = plt.barh(range(len(final_names)), final_importance, 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Remove value labels as requested
        # Remove y-label as requested
        plt.xlabel(r'Mean $|$SHAP Value$|$', fontsize=18, fontweight='bold')
        plt.yticks(range(len(final_names)))  # Use final_names instead of input_names
        plt.grid(True, alpha=0.3, axis='x')
        
        # Remove y-axis labels on the right plot to avoid duplication
        plt.gca().set_yticklabels([])
        
        # Increase tick label sizes to match left plot
        ax2.tick_params(axis='both', which='major', labelsize=16)  # Match left plot tick size
        ax2.tick_params(axis='x', which='major', labelsize=16)     # X-axis tick labels
        
        # Align the right plot y-axis with the left plot for perfect horizontal alignment
        # Set the same y-limits and y-tick positions as the left plot
        ax2.set_ylim(left_ylim)
        ax2.set_yticks(left_yticks)
        
        # Ensure the bars are positioned to align with the left plot's feature positions
        # The left plot has features at specific y-positions, we need to match those
        # Get the actual y-positions of features from the left plot
        left_feature_positions = []
        for i, txt in enumerate(ax1.get_yticklabels()):
            if txt.get_text():  # Only non-empty labels
                left_feature_positions.append(i)
        
        # Clear the previous bars and redraw with correct alignment
        ax2.clear()
        plt.sca(ax2)
        
        # Create horizontal bar plot with positions matching left plot
        if len(left_feature_positions) == len(final_names):
            bars = plt.barh(left_feature_positions, final_importance, 
                           color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.yticks(left_feature_positions)
        else:
            y_positions = np.linspace(left_ylim[0], left_ylim[1], len(final_names))
            bars = plt.barh(y_positions, final_importance, 
                           color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            plt.yticks(y_positions)
        
        # Reapply formatting
        plt.xlabel(r'Mean $|$SHAP Value$|$', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.gca().set_yticklabels([])
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.tick_params(axis='x', which='major', labelsize=16)
        
        # Remove top and right borders, keep only left and bottom
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(True)
        ax2.spines['bottom'].set_visible(True)
        
        # Set the same y-limits as left plot for perfect alignment
        ax2.set_ylim(left_ylim)
        
        # Add colorbar on the far right
        ax3 = fig.add_subplot(gs[0, 2])  # Use the third column for colorbar
        
        # Create colorbar using the custom colormap
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=ax3)
        cbar.set_label(r'Feature Value', rotation=90, labelpad=20, fontweight='bold', fontsize=18)

        # Set colorbar ticks to show High and Low labels with larger font
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.ax.tick_params(labelsize=16)  # Increase Low/High font size to match other elements
        
        plt.tight_layout()
        
        # Save combined SHAP plot in both PNG and PDF formats
        clean_output_name = output_name.replace('/', '_').replace(' ', '_')
        plt.savefig(f"{output_dir}/shap_analysis_{clean_output_name}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f"{output_dir}/shap_analysis_{clean_output_name}.pdf", 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Print SHAP analysis results - adapted for grouping
        print(f"\n" + "="*60)
        print(f"SHAP ANALYSIS RESULTS: {output_name}")
        print("="*60)
        print("Feature Importance Rankings (Mean |SHAP Value|):")
        
        # Print features in importance order (from right plot)
        for i, (name, importance) in enumerate(zip(final_names, final_importance)):
            print(f"{i+1:2d}. {name:30}: {importance:.6f}")
        
        # If there are grouped features, also print details
        if grouped_pattern_found and grouped_features_count > 0:
            print(f"\nDetailed breakdown of grouped features:")
            for i, (orig_name, orig_importance) in enumerate(zip(grouped_names, grouped_importance_values)):
                print(f"    {orig_name:25}: {orig_importance:.6f}")
            print(f"    → Average of grouped features: {grouped_importance_mean:.6f}")
        
    
    print(f"\n" + "="*80)
    print(f"SHAP analysis complete! Generated plots:")
    print(f"   - SHAP summary plots (beeswarm)")
    print(f"   - SHAP feature importance plots")
    print(f"   - SHAP data files")
    print(f"="*80)