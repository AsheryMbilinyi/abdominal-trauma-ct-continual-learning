"""
Configuration file for RSNA Continual Learning experiments.
Modify these parameters to customize your experiments.
"""

import torch

# ================================
# Data Configuration
# ================================

DATA_CONFIG = {
    # Path to dataset root directory
    'data_root': '/path/to/rsna/dataset',  # Update if not using Kaggle

    # Optional Kaggle dataset id (used with kagglehub)
    'kaggle_dataset': None,

    # Data splits
    'train_split': 'train',
    'test_split': 'test',

    # File format ('dcm' for DICOM, 'png' for PNG images)
    'file_extension': 'png',

    # Image preprocessing
    'target_size': (256, 256),
    'normalize': True,

    # CT Windowing parameters
    'windows': {
        'brain_subdural': {'width': 2, 'level': 1},
        'bone': {'width': 2048, 'level': 1},
        'soft_tissue': {'width': 300, 'level': 150}
    }
}

# ================================
# Model Configuration
# ================================

MODEL_CONFIG = {
    # Number of classes in dataset
    'num_classes': 2,

    # Input image channels (3 for multi-window CT)
    'input_channels': 3,

    # Dropout rate
    'dropout_rate': 0.5,

    # Model architecture (can be extended)
    'architecture': 'cnn',  # 'cnn', 'resnet', 'densenet'
}

# ================================
# Training Configuration
# ================================

TRAINING_CONFIG = {
    # Basic training parameters
    'learning_rate': 0.001,
    'batch_size': 12,  # Optimized for RTX 4050 (6GB VRAM)
    'iters_per_task': 500,

    # Optimizer settings
    'optimizer': 'adam',
    'betas': (0.9, 0.999),
    'weight_decay': 0.0,

    # Mixed precision (RTX 4050 optimization)
    'use_amp': True,  # Automatic Mixed Precision
    'gradient_accumulation_steps': 1,

    # DataLoader settings (Windows optimization)
    'num_workers': 2,  # Works best on Windows
    'pin_memory': True,
    'persistent_workers': False,

    # Learning rate scheduler
    'use_scheduler': False,
    'scheduler_type': 'step',  # 'step', 'cosine', 'plateau'
    'scheduler_params': {
        'step_size': 100,
        'gamma': 0.1
    },

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Reproducibility
    'random_seed': 42,

    # Logging
    'log_interval': 50,
    'save_checkpoints': True,
    'checkpoint_dir': './checkpoints'
}

# ================================
# Continual Learning Configuration
# ================================

CONTINUAL_LEARNING_CONFIG = {
    # Task setup
    'num_tasks': 2,
    'task_type': 'class_incremental',  # 'class_incremental', 'domain_incremental'
    'classes_per_task': None,  # None for automatic split

    # EWC parameters
    'ewc': {
        'lambda': 5000.0,  # Regularization strength
        'gamma': 1.0,      # Running average parameter for online EWC
        'fisher_samples': 150,  # Reduced for RTX 4050 (faster computation)
        'fisher_batch_size': 1,
    },

    # Experience Replay parameters
    'replay': {
        'buffer_size_per_class': 40,  # Reduced for RTX 4050 (6GB VRAM)
        'selection_strategy': 'random',  # 'random', 'herding', 'cluster'
        'update_strategy': 'reservoir',  # 'reservoir', 'ring_buffer'
    },

    # Combined method parameters
    'combined': {
        'ewc_lambda': 5000.0,
        'buffer_size_per_class': 40,  # Reduced for RTX 4050 (6GB VRAM)
        'ewc_weight': 0.5,  # Relative importance of EWC vs replay
    }
}

# ================================
# Evaluation Configuration
# ================================

EVALUATION_CONFIG = {
    # Test batch size (can be larger than training)
    'batch_size': 32,

    # Maximum samples to evaluate (None for all)
    'max_test_samples': None,

    # Metrics to compute
    'metrics': ['accuracy', 'forgetting', 'forward_transfer', 'backward_transfer'],

    # Save predictions
    'save_predictions': False,
    'predictions_dir': './predictions'
}

# ================================
# Visualization Configuration
# ================================

VISUALIZATION_CONFIG = {
    # Figure settings
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'seaborn',

    # Save figures
    'save_figures': True,
    'figures_dir': './figures',
    'figure_format': 'png',  # 'png', 'pdf', 'svg'

    # Plot settings
    'show_plots': True,
    'color_palette': 'Set2',
}

# ================================
# Experiment Tracking
# ================================

EXPERIMENT_CONFIG = {
    # Experiment name
    'experiment_name': 'rsna_continual_learning',

    # Results directory
    'results_dir': './results',

    # Save results
    'save_results': True,
    'results_format': 'csv',  # 'csv', 'json', 'pickle'

    # Wandb integration (optional)
    'use_wandb': False,
    'wandb_project': 'rsna-continual-learning',
    'wandb_entity': None,
}

# ================================
# Quick Presets
# ================================

# Fast debugging preset (small model, few iterations)
DEBUG_PRESET = {
    'iters_per_task': 50,
    'batch_size': 8,
    'fisher_samples': 20,
    'buffer_size_per_class': 10,
}

# High-quality preset (longer training, larger buffer)
HIGHQUALITY_PRESET = {
    'iters_per_task': 2000,
    'batch_size': 32,
    'fisher_samples': 500,
    'buffer_size_per_class': 200,
    'learning_rate': 0.0005,
}

# Memory-efficient preset (smaller batches, smaller buffer)
MEMORY_EFFICIENT_PRESET = {
    'batch_size': 4,
    'buffer_size_per_class': 20,
    'fisher_batch_size': 1,
}

# RTX 4050 optimized preset (6GB VRAM, Windows)
RTX4050_PRESET = {
    'batch_size': 12,
    'iters_per_task': 500,
    'fisher_samples': 150,
    'buffer_size_per_class': 40,
    'use_amp': True,
    'num_workers': 2,
    'pin_memory': True,
}

# ================================
# Helper Functions
# ================================

def get_config(preset=None):
    """
    Get complete configuration dictionary.

    Args:
        preset: Optional preset name ('debug', 'highquality', 'memory_efficient', 'rtx4050')

    Returns:
        dict: Complete configuration
    """
    config = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'continual_learning': CONTINUAL_LEARNING_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
    }

    # Apply preset if specified
    if preset == 'debug':
        config['training'].update(DEBUG_PRESET)
        config['continual_learning']['ewc']['fisher_samples'] = DEBUG_PRESET['fisher_samples']
        config['continual_learning']['replay']['buffer_size_per_class'] = DEBUG_PRESET['buffer_size_per_class']
    elif preset == 'highquality':
        config['training'].update(HIGHQUALITY_PRESET)
        config['continual_learning']['ewc']['fisher_samples'] = HIGHQUALITY_PRESET['fisher_samples']
        config['continual_learning']['replay']['buffer_size_per_class'] = HIGHQUALITY_PRESET['buffer_size_per_class']
    elif preset == 'memory_efficient':
        config['training'].update(MEMORY_EFFICIENT_PRESET)
        config['continual_learning']['replay']['buffer_size_per_class'] = MEMORY_EFFICIENT_PRESET['buffer_size_per_class']
    elif preset == 'rtx4050':
        config['training'].update(RTX4050_PRESET)
        config['continual_learning']['ewc']['fisher_samples'] = RTX4050_PRESET['fisher_samples']
        config['continual_learning']['replay']['buffer_size_per_class'] = RTX4050_PRESET['buffer_size_per_class']

    return config


def print_config(config):
    """Print configuration in a readable format."""
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)

    for section, params in config.items():
        print(f"\n{section.upper().replace('_', ' ')}:")
        print("-" * 40)
        _print_dict(params, indent=2)

    print("\n" + "="*60 + "\n")


def _print_dict(d, indent=0):
    """Helper to print nested dictionaries."""
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            _print_dict(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print_config(config)

    print("\nTo use debug preset:")
    print("config = get_config(preset='debug')")

    print("\nTo modify specific parameters:")
    print("config['training']['learning_rate'] = 0.0001")
