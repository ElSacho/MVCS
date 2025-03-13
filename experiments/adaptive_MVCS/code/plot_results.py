import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_and_print_results_one_file(file_path):
    pkl_file = os.path.basename(file_path)
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
            if isinstance(results, dict):
                print(f"Contents of {pkl_file}:")
                mean_results = {}
                for key in results[0].keys():
                    # Extract values for the current key across experiments
                    values = [results[exp][key] for exp in results]
                    
                    # Remove the minimum and maximum values
                    values_sorted = sorted(values)
                    values_trimmed = values_sorted[1:-1]  # Exclude first (min) and last (max)
                    
                    # Calculate the mean of the remaining values
                    mean_results[key] = np.mean(values_trimmed)

                for key, value in mean_results.items():
                    print(f"{key:>50}: {value:.6f}")
            else:
                print(f"Warning: {pkl_file} does not contain a dictionary.")
    except Exception as e:
        print(f"Failed to load {pkl_file}: {e}")
    print("\n\n")


def load_and_print_results(folder_path):
    """
    Load and print the contents of all .pkl files in the specified folder.

    Parameters:
        folder_path (str): Path to the folder containing .pkl files.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # List all .pkl files in the folder
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl') and 'tab' not in f]

    if not pkl_files:
        print("No .pkl files found in the folder.")
        return

    # Iterate through each .pkl file, load and print its content
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        load_and_print_results_one_file(file_path)



def load_and_plot_results_averaged_one_file(file_path):
    pkl_file = os.path.basename(file_path)
    print(f"Contents of {pkl_file}:")
    try:
        with open(file_path, 'rb') as f:
            tab_results = pickle.load(f)
            if isinstance(tab_results, dict):    
                # Compute average results
                num_experiments = len(tab_results)
                keys = list(tab_results.values())[0].keys()

                average_results = {key: np.mean([tab_results[exp][key] for exp in tab_results], axis=0) for key in keys}

                # Plot average results
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                # First plot: Warm start train and calibration losses
                axes[0].plot(average_results["warm_start_train_loss"], label="Warm Start Train Loss")
                axes[0].plot(average_results["warm_start_calibration_loss"], label="Warm Start Calibration Loss")
                axes[0].set_title("Average: Warm Start Losses")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend()

                # Second plot: Train and calibration losses (our method)
                axes[1].plot(average_results["train_our_loss"], label="Train Our Loss")
                axes[1].plot(average_results["calibration_our_loss"], label="Calibration Our Loss")
                axes[1].set_title("Average: Our Method Losses")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Loss")
                axes[1].legend()

                # Third plot: q values
                axes[2].plot(average_results["q"], label="q", marker="o")
                axes[2].set_title("Average: q Values")
                axes[2].set_xlabel("Epoch")
                axes[2].set_ylabel("q")
                axes[2].legend()

                plt.tight_layout()
                plt.show()
            else:
                print(f"Warning: {pkl_file} does not contain a dictionary.")
    except Exception as e:
        print(f"Failed to load {pkl_file}: {e}")
    print("\n\n")

def load_and_plot_results_averaged(folder_path):
    """
    Load and print the contents of all .pkl files in the specified folder.

    Parameters:
        folder_path (str): Path to the folder containing .pkl files.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # List all .pkl files in the folder
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl') and 'tab' in f]

    if not pkl_files:
        print("No .pkl files found in the folder.")
        return

    # Iterate through each .pkl file, load and print its content
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        load_and_plot_results_averaged_one_file(file_path)



def load_and_plot_results_one_by_one_one_file(file_path):
    pkl_file = os.path.basename(file_path)
    print(f"Contents of {pkl_file}:")
    try:
        with open(file_path, 'rb') as f:
            tab_results = pickle.load(f)
            if isinstance(tab_results, dict):
                for experiment, results in tab_results.items():
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                    # First plot: Warm start train and calibration losses
                    axes[0].plot(results["warm_start_train_loss"], label="Warm Start Train Loss")
                    axes[0].plot(results["warm_start_calibration_loss"], label="Warm Start Calibration Loss")
                    axes[0].set_title(f"{experiment}: Warm Start Losses")
                    axes[0].set_xlabel("Epoch")
                    axes[0].set_ylabel("Loss")
                    axes[0].legend()

                    # Second plot: Train and calibration losses (our method)
                    axes[1].plot(results["train_our_loss"], label="Train Our Loss")
                    axes[1].plot(results["calibration_our_loss"], label="Calibration Our Loss")
                    axes[1].set_title(f"{experiment}: Our Method Losses")
                    axes[1].set_xlabel("Epoch")
                    axes[1].set_ylabel("Loss")
                    axes[1].legend()

                    # Third plot: q values
                    axes[2].plot(results["q"], label="q", marker="o")
                    axes[2].set_title(f"{experiment}: q Values")
                    axes[2].set_xlabel("Epoch")
                    axes[2].set_ylabel("q")
                    axes[2].legend()

                    plt.tight_layout()
                    plt.show()

                # Compute average results
                num_experiments = len(tab_results)
                keys = list(tab_results.values())[0].keys()

                average_results = {key: np.mean([tab_results[exp][key] for exp in tab_results], axis=0) for key in keys}

                # Plot average results
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                # First plot: Warm start train and calibration losses
                axes[0].plot(average_results["warm_start_train_loss"], label="Warm Start Train Loss")
                axes[0].plot(average_results["warm_start_calibration_loss"], label="Warm Start Calibration Loss")
                axes[0].set_title("Average: Warm Start Losses")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend()

                # Second plot: Train and calibration losses (our method)
                axes[1].plot(average_results["train_our_loss"], label="Train Our Loss")
                axes[1].plot(average_results["calibration_our_loss"], label="Calibration Our Loss")
                axes[1].set_title("Average: Our Method Losses")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Loss")
                axes[1].legend()

                # Third plot: q values
                axes[2].plot(average_results["q"], label="q", marker="o")
                axes[2].set_title("Average: q Values")
                axes[2].set_xlabel("Epoch")
                axes[2].set_ylabel("q")
                axes[2].legend()

                plt.tight_layout()
                plt.show()
            else:
                print(f"Warning: {pkl_file} does not contain a dictionary.")
    except Exception as e:
        print(f"Failed to load {pkl_file}: {e}")
    print("\n\n")



def load_and_plot_results_one_by_one(folder_path):
    """
    Load and print the contents of all .pkl files in the specified folder.

    Parameters:
        folder_path (str): Path to the folder containing .pkl files.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # List all .pkl files in the folder
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl') and 'tab' in f]

    if not pkl_files:
        print("No .pkl files found in the folder.")
        return

    # Iterate through each .pkl file, load and print its content
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        load_and_plot_results_one_by_one_one_file(file_path)