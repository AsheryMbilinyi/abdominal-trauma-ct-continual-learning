"""
Utility functions for RSNA Continual Learning project.
These functions can be imported and reused across different scripts.
"""

import os
import json
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


# ================================
# Reproducibility
# ================================

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================
# File and Directory Management
# ================================

def create_directories(dirs):
    """
    Create directories if they don't exist.

    Args:
        dirs: List of directory paths or single path string
    """
    if isinstance(dirs, str):
        dirs = [dirs]

    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def download_kagglehub_dataset(dataset_id):
    """
    Download a dataset via kagglehub and return the local path.

    Args:
        dataset_id: Kaggle dataset identifier, e.g. "ashery/rsna-2023-abdominal-trauma-processed-dataset"
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for dataset download. "
            "Install it with: pip install kagglehub"
        ) from exc

    return kagglehub.dataset_download(dataset_id)


def resolve_data_root(data_root=None, kaggle_dataset=None):
    """
    Resolve the dataset root path from a local path or Kaggle dataset id.

    Args:
        data_root: Local path to dataset root directory.
        kaggle_dataset: Kaggle dataset id to download via kagglehub.
    """
    if kaggle_dataset:
        return download_kagglehub_dataset(kaggle_dataset)

    if not data_root or data_root == '/path/to/rsna/dataset':
        raise ValueError("Please provide a valid data_root or kaggle_dataset.")

    return data_root


def save_checkpoint(model, optimizer, epoch, filepath, **kwargs):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch/iteration
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load model on

    Returns:
        dict: Checkpoint dictionary with additional metadata
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {filepath}")

    return checkpoint


# ================================
# Results Saving and Loading
# ================================

def save_results(results, filepath, format='json'):
    """
    Save experimental results.

    Args:
        results: Results dictionary
        filepath: Path to save file
        format: Format to save ('json', 'pickle', 'csv')
    """
    filepath = Path(filepath)
    create_directories(filepath.parent)

    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = _make_json_serializable(results)
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

    elif format == 'csv':
        import pandas as pd
        # Flatten results if needed
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)

    print(f"Results saved to {filepath}")


def load_results(filepath):
    """
    Load experimental results.

    Args:
        filepath: Path to results file

    Returns:
        Results dictionary or DataFrame
    """
    filepath = Path(filepath)

    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)

    elif filepath.suffix == '.pkl' or filepath.suffix == '.pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    elif filepath.suffix == '.csv':
        import pandas as pd
        return pd.read_csv(filepath)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def _make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


# ================================
# Experiment Logging
# ================================

class ExperimentLogger:
    """
    Logger for tracking experiments and results.
    """

    def __init__(self, experiment_name, log_dir='./logs'):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        create_directories(self.log_dir)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        self.metrics = {}

    def log(self, message):
        """Log a message to file and print."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        print(log_message)

        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_metric(self, name, value, step=None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        if name not in self.metrics:
            self.metrics[name] = []

        metric_entry = {'value': value}
        if step is not None:
            metric_entry['step'] = step

        self.metrics[name].append(metric_entry)

        log_msg = f"Metric - {name}: {value}"
        if step is not None:
            log_msg += f" (step {step})"

        self.log(log_msg)

    def save_metrics(self, filepath=None):
        """Save logged metrics to file."""
        if filepath is None:
            filepath = self.log_dir / f"{self.experiment_name}_metrics.json"

        save_results(self.metrics, filepath, format='json')


# ================================
# Data Statistics
# ================================

def compute_dataset_statistics(dataset):
    """
    Compute statistics for a dataset.

    Args:
        dataset: PyTorch Dataset

    Returns:
        dict: Statistics including class distribution, mean, std
    """
    from collections import Counter
    from torch.utils.data import DataLoader

    # Class distribution
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)

    # Compute mean and std (sample-based for efficiency)
    sample_size = min(1000, len(dataset))
    indices = random.sample(range(len(dataset)), sample_size)

    loader = DataLoader(dataset, batch_size=32, sampler=indices)

    all_images = []
    for images, _ in loader:
        all_images.append(images)

    all_images = torch.cat(all_images, dim=0)

    stats = {
        'num_samples': len(dataset),
        'num_classes': len(class_counts),
        'class_distribution': dict(class_counts),
        'mean': all_images.mean(dim=(0, 2, 3)).tolist(),
        'std': all_images.std(dim=(0, 2, 3)).tolist(),
    }

    return stats


def print_dataset_info(dataset, name='Dataset'):
    """
    Print dataset information.

    Args:
        dataset: PyTorch Dataset
        name: Name to display
    """
    stats = compute_dataset_statistics(dataset)

    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Class distribution: {stats['class_distribution']}")
    print(f"Image mean (per channel): {[f'{m:.4f}' for m in stats['mean']]}")
    print(f"Image std (per channel): {[f'{s:.4f}' for s in stats['std']]}")
    print(f"{'='*50}\n")


# ================================
# Model Utilities
# ================================

def count_parameters(model):
    """
    Count trainable and total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        tuple: (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return trainable, total


def get_model_size_mb(model):
    """
    Get approximate model size in MB.

    Args:
        model: PyTorch model

    Returns:
        float: Model size in MB
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    size_mb = (param_size + buffer_size) / (1024 ** 2)

    return size_mb


def print_model_summary(model):
    """
    Print comprehensive model summary.

    Args:
        model: PyTorch model
    """
    trainable, total = count_parameters(model)
    size_mb = get_model_size_mb(model)

    print(f"\n{'='*50}")
    print("Model Summary")
    print(f"{'='*50}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"{'='*50}\n")


# ================================
# Continual Learning Metrics
# ================================

def calculate_average_accuracy(accuracy_matrix):
    """
    Calculate average accuracy across all tasks.

    Args:
        accuracy_matrix: 2D numpy array of accuracies

    Returns:
        float: Average accuracy
    """
    return accuracy_matrix[-1, :].mean()


def calculate_forgetting(accuracy_matrix):
    """
    Calculate average forgetting across tasks.

    Args:
        accuracy_matrix: 2D numpy array where element [i,j] is
                        accuracy on task j after training on task i

    Returns:
        float: Average forgetting
    """
    num_tasks = accuracy_matrix.shape[0]
    forgetting_values = []

    for task_id in range(num_tasks - 1):
        # Maximum accuracy achieved on this task
        max_acc = accuracy_matrix[task_id, task_id]
        # Final accuracy on this task
        final_acc = accuracy_matrix[-1, task_id]
        # Forgetting for this task
        forgetting = max_acc - final_acc
        forgetting_values.append(forgetting)

    return np.mean(forgetting_values) if forgetting_values else 0.0


def calculate_forward_transfer(accuracy_matrix, random_baseline):
    """
    Calculate forward transfer (zero-shot performance on future tasks).

    Args:
        accuracy_matrix: 2D numpy array of accuracies
        random_baseline: Random guess accuracy (e.g., 1/num_classes * 100)

    Returns:
        float: Average forward transfer
    """
    num_tasks = accuracy_matrix.shape[0]
    transfer_values = []

    for task_id in range(1, num_tasks):
        # Accuracy on task before training on it
        before_training = accuracy_matrix[task_id - 1, task_id]
        # Transfer compared to random baseline
        transfer = before_training - random_baseline
        transfer_values.append(transfer)

    return np.mean(transfer_values) if transfer_values else 0.0


def calculate_backward_transfer(accuracy_matrix):
    """
    Calculate backward transfer (influence on previous tasks).

    Args:
        accuracy_matrix: 2D numpy array of accuracies

    Returns:
        float: Average backward transfer
    """
    num_tasks = accuracy_matrix.shape[0]
    transfer_values = []

    for task_id in range(num_tasks - 1):
        # Accuracy right after training on task
        immediately_after = accuracy_matrix[task_id, task_id]
        # Final accuracy on task
        final = accuracy_matrix[-1, task_id]
        # Backward transfer
        transfer = final - immediately_after
        transfer_values.append(transfer)

    return np.mean(transfer_values) if transfer_values else 0.0


def compute_all_metrics(accuracy_matrix, num_classes):
    """
    Compute all continual learning metrics.

    Args:
        accuracy_matrix: 2D numpy array of accuracies
        num_classes: Number of classes (for random baseline)

    Returns:
        dict: All metrics
    """
    random_baseline = 100.0 / num_classes

    metrics = {
        'average_accuracy': calculate_average_accuracy(accuracy_matrix),
        'forgetting': calculate_forgetting(accuracy_matrix),
        'forward_transfer': calculate_forward_transfer(accuracy_matrix, random_baseline),
        'backward_transfer': calculate_backward_transfer(accuracy_matrix),
    }

    return metrics


def print_metrics_summary(metrics_dict, method_names=None):
    """
    Print comparison of metrics across methods.

    Args:
        metrics_dict: Dictionary mapping method names to accuracy matrices
        method_names: Optional list of method names to display
    """
    if method_names is None:
        method_names = list(metrics_dict.keys())

    print(f"\n{'='*80}")
    print("CONTINUAL LEARNING METRICS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Avg Acc':<12} {'Forgetting':<12} {'Fwd Transfer':<14} {'Bwd Transfer':<12}")
    print(f"{'-'*80}")

    for method in method_names:
        matrix = metrics_dict[method]
        # Assuming binary classification for random baseline
        num_classes = 2
        metrics = compute_all_metrics(matrix, num_classes)

        print(f"{method:<25} "
              f"{metrics['average_accuracy']:>6.2f}%      "
              f"{metrics['forgetting']:>6.2f}%      "
              f"{metrics['forward_transfer']:>6.2f}%        "
              f"{metrics['backward_transfer']:>6.2f}%")

    print(f"{'='*80}\n")


# ================================
# Device Management
# ================================

def get_device(device_id=None):
    """
    Get PyTorch device.

    Args:
        device_id: Optional GPU device ID

    Returns:
        torch.device: Device object
    """
    if torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    return device


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("No GPU available.")


# ================================
# Main
# ================================

if __name__ == "__main__":
    print("Utility functions for RSNA Continual Learning")
    print("\nAvailable functions:")
    print("- set_seed()")
    print("- save/load_checkpoint()")
    print("- save/load_results()")
    print("- ExperimentLogger")
    print("- compute_dataset_statistics()")
    print("- count_parameters()")
    print("- calculate_*_metrics()")
    print("- get_device()")
