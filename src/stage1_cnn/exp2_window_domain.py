# ==============================================================================
# RSNA Continual Learning Experiment - Version 2
# 3-Task Window-Based Continual Learning
# ==============================================================================
# Tasks are defined by CT window types:
#   Task 1: Brain Window (WL=40, WW=80)
#   Task 2: Lung Window (WL=-600, WW=1500)
#   Task 3: Soft Tissue Window (WL=50, WW=400)
# ==============================================================================

# Standard libraries
import os
import numpy as np
import pandas as pd
import copy
import tqdm
from pathlib import Path
import random
from collections import Counter
from datetime import datetime
import logging
import time

# Image processing
import cv2
from PIL import Image

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

FORCE_CPU = os.environ.get('FORCE_CPU', '0') == '1'
FORCE_CUDA = os.environ.get('FORCE_CUDA', '0') == '1'

if os.environ.get('DEBUG_RUN', '0') == '1':
    plt.switch_backend('Agg')

# ============================================================================
# LOGGING SETUP - Save all output to file
# ============================================================================
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = LOG_DIR / f'experiment_v2_windows_{timestamp}.log'

# Setup file-only logger (console output via print for immediate feedback)
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def log(msg):
    """Print to console immediately AND save to log file."""
    print(msg, flush=True)
    logger.info(msg)

# Start timing the entire experiment
EXPERIMENT_START_TIME = time.time()

print("=" * 70, flush=True)
print("RSNA Continual Learning - 3-Task CT Window Experiment (V2)", flush=True)
print("=" * 70, flush=True)
log(f"Timestamp: {timestamp}")
log(f"Log file: {LOG_FILE}")
log(f"Results will be saved to: logs/results_v2_windows_{timestamp}.csv")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (not FORCE_CPU) and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(42)

# Local data configuration
DATA_DIR = Path('data')
DATA_ROOT = DATA_DIR / 'RSNA2023ProcessedImages'
LABELS_CSV = DATA_DIR / 'train.csv'
DEBUG_RUN = os.environ.get('DEBUG_RUN', '0') == '1'

# ============================================================================
# DATA SAMPLING CONFIGURATION
# ============================================================================
MAX_PATIENTS_PER_CLASS = 4 if DEBUG_RUN else 100
MAX_IMAGES_PER_PATIENT = 5 if DEBUG_RUN else 20

# Device configuration
if FORCE_CUDA and FORCE_CPU:
    print("FORCE_CUDA=1 overrides FORCE_CPU=1; using CUDA if available.")
    FORCE_CPU = False

if FORCE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("FORCE_CUDA=1 but CUDA is not available.")

device = torch.device('cuda' if (FORCE_CUDA or ((not FORCE_CPU) and torch.cuda.is_available())) else 'cpu')
AMP_ENABLED = device.type == 'cuda'
AMP_DEVICE = 'cuda' if device.type == 'cuda' else 'cpu'

log(f"Using device: {device}")
if AMP_ENABLED:
    log("Automatic Mixed Precision (AMP) enabled.")
if device.type == 'cuda':
    try:
        gpu_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        log(f"Active GPU: {gpu_name}")
        log(f"GPU total memory: {total_mem:.1f} GB")
    except Exception as exc:
        log(f"GPU info check failed: {exc}")

# ============================================================================
# CT WINDOWING FUNCTIONS
# ============================================================================
# CT windows highlight different tissue types by adjusting contrast
# WL = Window Level (center), WW = Window Width

def apply_ct_window(img, window_level, window_width):
    """Apply CT windowing to an image."""
    img = img.astype(np.float32)
    min_val = window_level - window_width // 2
    max_val = window_level + window_width // 2
    windowed = np.clip(img, min_val, max_val)
    windowed = ((windowed - min_val) / (max_val - min_val)) * 255.0
    return windowed.astype(np.uint8)

def apply_brain_window(img):
    """Brain window: WL=40, WW=80 - highlights brain tissue"""
    return apply_ct_window(img, window_level=40, window_width=80)

def apply_lung_window(img):
    """Lung window: WL=-600, WW=1500 - highlights lung/air"""
    return apply_ct_window(img, window_level=-600, window_width=1500)

def apply_soft_tissue_window(img):
    """Soft tissue window: WL=50, WW=400 - highlights soft tissues"""
    return apply_ct_window(img, window_level=50, window_width=400)

# Window type mapping for 3 tasks
CT_WINDOWS = {
    0: ('Brain', apply_brain_window),
    1: ('Lung', apply_lung_window),
    2: ('Soft Tissue', apply_soft_tissue_window),
}

WINDOW_NAMES = ['Brain', 'Lung', 'Soft Tissue']

def load_image_tensor(img_path, target_size, transform=None, window_type=None):
    """Load an image from disk and return a normalized tensor.

    Args:
        img_path: Path to image file
        target_size: (height, width) tuple
        transform: Optional transform to apply
        window_type: 0=Brain, 1=Lung, 2=Soft Tissue, None=no windowing
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load: {img_path}")

    # Apply CT windowing if specified
    if window_type is not None and window_type in CT_WINDOWS:
        _, window_func = CT_WINDOWS[window_type]
        img = window_func(img)

    # Resize if needed
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert grayscale to 3-channel for CNN compatibility
    img = np.stack([img, img, img], axis=-1)

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    if transform:
        img_tensor = transform(img_tensor)
    return img_tensor


class RSNAWindowDataset(Dataset):
    """
    Dataset for RSNA images with CT windowing support.

    For window-based continual learning:
    - window_type=0: Brain window
    - window_type=1: Lung window
    - window_type=2: Soft Tissue window
    """

    def __init__(self, image_paths, labels, target_size=(256, 256), transform=None, window_type=0):
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size
        self.transform = transform
        self.window_type = window_type

        assert len(image_paths) == len(labels), "Mismatch between images and labels"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img_tensor = load_image_tensor(
            img_path, self.target_size, self.transform,
            window_type=self.window_type
        )
        return img_tensor, label


class MemorySetDataset(Dataset):
    """Dataset from stored samples (for replay) with window type support."""

    def __init__(self, memory_sets, target_size, transform=None):
        """
        Args:
            memory_sets: List of (image_path, label, window_type) tuples
            target_size: Image size tuple
            transform: Optional transform
        """
        self.memory_sets = memory_sets
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.memory_sets)

    def __getitem__(self, idx):
        image_path, label, window_type = self.memory_sets[idx]
        img_tensor = load_image_tensor(
            image_path, self.target_size, self.transform,
            window_type=window_type
        )
        return img_tensor, label


def load_rsna_data(data_root, labels_csv, test_split=0.2, max_patients_per_class=None,
                   max_images_per_patient=None, seed=42):
    """Load RSNA dataset from local folder layout."""
    data_root = Path(data_root)
    labels_csv = Path(labels_csv)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    if 'patient_id' not in df.columns or 'any_injury' not in df.columns:
        raise ValueError("train.csv must include patient_id and any_injury columns")

    # Map patient_id -> label
    patient_labels = {
        int(row['patient_id']): int(row['any_injury'])
        for _, row in df[['patient_id', 'any_injury']].dropna().iterrows()
    }

    # Split patients by label
    label_to_patients = {0: [], 1: []}
    for pid, label in patient_labels.items():
        label_to_patients.setdefault(label, []).append(pid)

    rng = random.Random(seed)
    train_patients, test_patients = [], []

    for label, patients in label_to_patients.items():
        rng.shuffle(patients)
        if max_patients_per_class is not None:
            patients = patients[:max_patients_per_class]

        split_idx = int(len(patients) * (1.0 - test_split))
        train_patients.extend(patients[:split_idx])
        test_patients.extend(patients[split_idx:])

    def collect_images(patient_ids, split_name):
        image_paths, labels = [], []
        for pid in patient_ids:
            patient_dir = data_root / str(pid)
            if not patient_dir.exists():
                continue

            patient_images = []
            for series_dir in patient_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                for img_path in series_dir.glob('*.png'):
                    patient_images.append(img_path)
                    if max_images_per_patient and len(patient_images) >= max_images_per_patient:
                        break
                if max_images_per_patient and len(patient_images) >= max_images_per_patient:
                    break

            label = patient_labels.get(pid)
            if label is None:
                continue

            image_paths.extend([str(p) for p in patient_images])
            labels.extend([label] * len(patient_images))

        return image_paths, labels

    train_paths, train_labels = collect_images(train_patients, "train")
    test_paths, test_labels = collect_images(test_patients, "test")

    if not train_paths or not test_paths:
        raise ValueError('Train/test split produced no images.')

    log(f"\nLoaded {len(train_paths)} training images")
    log(f"Loaded {len(test_paths)} test images")
    log(f"Class distribution (train): {Counter(train_labels)}")

    return train_paths, train_labels, test_paths, test_labels


def create_window_task_datasets(image_paths, labels, target_size, num_tasks=3):
    """
    Create task datasets based on CT window types.

    Task 1: Brain Window
    Task 2: Lung Window
    Task 3: Soft Tissue Window

    Each task uses the SAME images but with DIFFERENT windowing applied.
    """
    task_datasets = []

    for task_id in range(num_tasks):
        window_name = CT_WINDOWS[task_id][0]
        task_dataset = RSNAWindowDataset(
            image_paths, labels, target_size=target_size, window_type=task_id
        )
        task_datasets.append(task_dataset)
        log(f"Task {task_id+1}: {window_name} Window, {len(task_dataset)} samples")

    return task_datasets


def fill_memory_buffer(dataset, buffer_size, window_type):
    """Fill memory buffer with samples from a task (window type)."""
    memory_sets = []

    indices = list(range(len(dataset)))
    if len(indices) > buffer_size:
        indices = random.sample(indices, buffer_size)

    for idx in indices:
        image = dataset.image_paths[idx]
        label = dataset.labels[idx]
        memory_sets.append((image, label, window_type))

    return memory_sets


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class MedicalImageClassifier(nn.Module):
    """CNN classifier for medical images."""

    def __init__(self, num_classes=2, input_channels=3, dropout_rate=0.5):
        super(MedicalImageClassifier, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.feature_size = 256 * 16 * 16

        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, self.feature_size)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_baseline(model, dataset, iters, lr, batch_size, device):
    """Standard training without continual learning strategies."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)

    model.train()
    model.to(device)
    log(f"  Training samples: {len(dataset)} | Batch size: {batch_size} | Iters: {iters}")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True, pin_memory=True)

    iters_left = 0
    progress_bar = tqdm.tqdm(range(1, iters + 1), desc="Training")

    for batch_idx in progress_bar:
        if iters_left == 0:
            data_iter = iter(data_loader)
            iters_left = len(data_loader)

        images, labels = next(data_iter)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        iters_left -= 1

        optimizer.zero_grad()

        with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})


def evaluate_accuracy(model, dataset, device, batch_size, num_workers=0):
    """Evaluate model accuracy."""
    model.eval()
    model.to(device)
    log(f"  Evaluating {len(dataset)} samples | Batch size: {batch_size}")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm.tqdm(data_loader, desc="  Eval", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def estimate_fisher_information(model, dataset, n_samples, device):
    """Estimate Fisher Information Matrix."""
    fisher_info = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_info[name.replace('.', '__')] = torch.zeros_like(param.data)

    model.eval()
    model.to(device)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    n_samples_processed = 0
    for images, labels in data_loader:
        if n_samples_processed >= n_samples:
            break

        images = images.to(device)

        model.zero_grad()
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        for c in range(outputs.size(1)):
            model.zero_grad()
            loss = -torch.log(probs[0, c] + 1e-10)
            loss.backward(retain_graph=True)

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name.replace('.', '__')] += (
                        probs[0, c].item() * param.grad.data.pow(2)
                    )

        n_samples_processed += 1

    for name in fisher_info:
        fisher_info[name] /= n_samples_processed

    return fisher_info


def train_with_ewc(model, dataset, iters, lr, batch_size, device, current_task, ewc_lambda=5000.0):
    """Train with EWC regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)

    model.train()
    model.to(device)
    log(f"  Training samples: {len(dataset)} | Batch size: {batch_size} | Iters: {iters}")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True, pin_memory=True)

    iters_left = 0
    progress_bar = tqdm.tqdm(range(1, iters + 1), desc=f"EWC Task {current_task}")

    for batch_idx in progress_bar:
        if iters_left == 0:
            data_iter = iter(data_loader)
            iters_left = len(data_loader)

        images, labels = next(data_iter)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        iters_left -= 1

        optimizer.zero_grad()

        with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if current_task > 1 and hasattr(model, 'fisher_info'):
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        name_normalized = name.replace('.', '__')
                        if name_normalized in model.fisher_info:
                            fisher = model.fisher_info[name_normalized].to(device)
                            optimal = model.optimal_params[name_normalized].to(device)
                            ewc_loss += (fisher * (param - optimal).pow(2)).sum()

                loss += (ewc_lambda / 2) * ewc_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})


def train_with_replay(model, dataset, iters, lr, batch_size, device, current_task, buffer_dataset=None):
    """Train with experience replay."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)

    model.train()
    model.to(device)
    log(f"  Training samples: {len(dataset)} | Batch size: {batch_size} | Iters: {iters}")
    if buffer_dataset is not None:
        log(f"  Replay samples: {len(buffer_dataset)}")

    current_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True, pin_memory=True)

    use_replay = buffer_dataset is not None and len(buffer_dataset) > 0
    if use_replay:
        replay_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True, pin_memory=True)

    iters_left_current = 0
    iters_left_replay = 0
    progress_bar = tqdm.tqdm(range(1, iters + 1), desc=f"Replay Task {current_task}")

    for batch_idx in progress_bar:
        if iters_left_current == 0:
            current_iter = iter(current_loader)
            iters_left_current = len(current_loader)

        images_current, labels_current = next(current_iter)
        images_current = images_current.to(device, non_blocking=True)
        labels_current = labels_current.to(device, non_blocking=True)
        iters_left_current -= 1

        if use_replay:
            if iters_left_replay == 0:
                replay_iter = iter(replay_loader)
                iters_left_replay = len(replay_loader)

            images_replay, labels_replay = next(replay_iter)
            images_replay = images_replay.to(device, non_blocking=True)
            labels_replay = labels_replay.to(device, non_blocking=True)
            iters_left_replay -= 1

            images = torch.cat([images_current, images_replay], dim=0)
            labels = torch.cat([labels_current, labels_replay], dim=0)
        else:
            images = images_current
            labels = labels_current

        optimizer.zero_grad()

        with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})


def train_with_ewc_replay(model, dataset, iters, lr, batch_size, device, current_task,
                         buffer_dataset=None, ewc_lambda=5000.0):
    """Train with both EWC and replay."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)

    model.train()
    model.to(device)
    log(f"  Training samples: {len(dataset)} | Batch size: {batch_size} | Iters: {iters}")
    if buffer_dataset is not None:
        log(f"  Replay samples: {len(buffer_dataset)}")

    current_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=True, pin_memory=True)

    use_replay = buffer_dataset is not None and len(buffer_dataset) > 0
    if use_replay:
        replay_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True, pin_memory=True)

    iters_left_current = 0
    iters_left_replay = 0
    progress_bar = tqdm.tqdm(range(1, iters + 1), desc=f"EWC+Replay Task {current_task}")

    for batch_idx in progress_bar:
        if iters_left_current == 0:
            current_iter = iter(current_loader)
            iters_left_current = len(current_loader)

        images_current, labels_current = next(current_iter)
        images_current = images_current.to(device, non_blocking=True)
        labels_current = labels_current.to(device, non_blocking=True)
        iters_left_current -= 1

        if use_replay:
            if iters_left_replay == 0:
                replay_iter = iter(replay_loader)
                iters_left_replay = len(replay_loader)

            images_replay, labels_replay = next(replay_iter)
            images_replay = images_replay.to(device, non_blocking=True)
            labels_replay = labels_replay.to(device, non_blocking=True)
            iters_left_replay -= 1

            images = torch.cat([images_current, images_replay], dim=0)
            labels = torch.cat([labels_current, labels_replay], dim=0)
        else:
            images = images_current
            labels = labels_current

        optimizer.zero_grad()

        with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if current_task > 1 and hasattr(model, 'fisher_info'):
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        name_normalized = name.replace('.', '__')
                        if name_normalized in model.fisher_info:
                            fisher = model.fisher_info[name_normalized].to(device)
                            optimal = model.optimal_params[name_normalized].to(device)
                            ewc_loss += (fisher * (param - optimal).pow(2)).sum()

                loss += (ewc_lambda / 2) * ewc_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})


# ============================================================================
# METRICS AND VISUALIZATION
# ============================================================================

def calculate_metrics(accuracy_matrix):
    """Calculate continual learning metrics."""
    num_tasks = accuracy_matrix.shape[0]

    avg_acc = accuracy_matrix[-1, :].mean()

    forgetting_values = []
    for task_id in range(num_tasks - 1):
        max_acc = accuracy_matrix[task_id, task_id]
        final_acc = accuracy_matrix[-1, task_id]
        forgetting = max_acc - final_acc
        forgetting_values.append(forgetting)

    avg_forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return {'avg_accuracy': avg_acc, 'forgetting': avg_forgetting}


def plot_forgetting_matrix(accuracy_matrix, method_name, task_names=None, save_path=None):
    """Plot forgetting analysis heatmap."""
    num_tasks = accuracy_matrix.shape[0]
    if task_names is None:
        task_names = [f'Task {i+1}' for i in range(num_tasks)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=task_names, yticklabels=task_names,
                vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'})
    plt.xlabel('Test Task (Window Type)', fontsize=12)
    plt.ylabel('After Training on Task', fontsize=12)
    plt.title(f'Forgetting Analysis: {method_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log(f"  Saved plot: {save_path}")
    plt.close()


def clear_gpu_memory():
    """Clear GPU memory between experiments."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

# Load data
log("\n" + "="*70)
log("Loading Data...")
log("="*70)

train_paths, train_labels, test_paths, test_labels = load_rsna_data(
    DATA_ROOT, LABELS_CSV,
    test_split=0.2,
    max_patients_per_class=MAX_PATIENTS_PER_CLASS,
    max_images_per_patient=MAX_IMAGES_PER_PATIENT,
    seed=42,
)

# Experiment hyperparameters
NUM_TASKS = 3  # Brain, Lung, Soft Tissue windows
NUM_CLASSES = 2  # Binary: injury / no injury
TARGET_SIZE = (256, 256)

# ============================================================================
# OPTIMIZED HYPERPARAMETERS (based on tutorial comparison analysis)
# ============================================================================
# Previous values were suboptimal:
#   - EWC lambda was 50x too high (5000 vs tutorial's 100)
#   - Learning rate was 10x too low (0.001 vs tutorial's 0.01)
#   - Batch size was 8x too small (16 vs tutorial's 128)
# These optimizations should significantly improve EWC performance

# Increased iterations for better convergence with larger batch size
# With batch_size=32 and ~3200 samples, one epoch = ~100 iterations
# 1000 iterations = ~10 epochs per task
ITERS_PER_TASK = 1000 if not DEBUG_RUN else 1

# Increased learning rate closer to tutorial (0.01), but slightly lower
# for medical imaging which benefits from more conservative updates
LEARNING_RATE = 0.005

# Increased batch size for more stable gradients (important for EWC)
# RTX 4050 has 6GB VRAM, batch_size=32 should fit with 256x256 images
BATCH_SIZE = 32 if not DEBUG_RUN else 2
EVAL_BATCH_SIZE = 64 if not DEBUG_RUN else 2

# CRITICAL FIX: Reduced EWC lambda from 5000 to 100 (matching tutorial)
# High lambda was causing EWC regularization to dominate, preventing learning
EWC_LAMBDA = 100.0

# Increased replay buffer for better retention of old task knowledge
# More samples = better representation of previous task distribution
BUFFER_SIZE_PER_TASK = 500 if not DEBUG_RUN else 5

# Increased Fisher samples for more accurate importance estimation
FISHER_SAMPLES = 500 if not DEBUG_RUN else 5

log("\n" + "="*70)
log("Experiment Configuration (3-Task CT Window) - OPTIMIZED")
log("="*70)
log(f"  Tasks: {NUM_TASKS} (Brain, Lung, Soft Tissue windows)")
log(f"  Classes: {NUM_CLASSES} (injury / no injury)")
log(f"  Iterations/task: {ITERS_PER_TASK}")
log(f"  Learning rate: {LEARNING_RATE}")
log(f"  Batch size: {BATCH_SIZE}")
log(f"  Eval batch size: {EVAL_BATCH_SIZE}")
log(f"  EWC lambda: {EWC_LAMBDA}")
log(f"  Fisher samples: {FISHER_SAMPLES}")
log(f"  Replay buffer/task: {BUFFER_SIZE_PER_TASK}")
log(f"  Max patients/class: {MAX_PATIENTS_PER_CLASS}")
log(f"  Max images/patient: {MAX_IMAGES_PER_PATIENT}")

# Create task datasets (same images, different windows)
log("\nCreating window-based task datasets...")
train_task_datasets = create_window_task_datasets(train_paths, train_labels, TARGET_SIZE, NUM_TASKS)
test_task_datasets = create_window_task_datasets(test_paths, test_labels, TARGET_SIZE, NUM_TASKS)

# Model info
model = MedicalImageClassifier(num_classes=NUM_CLASSES, input_channels=3)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"\nModel created with {total_params:,} trainable parameters")
del model

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

# BASELINE EXPERIMENT
log("\n" + "="*70)
log("BASELINE EXPERIMENT (No Continual Learning)")
log("="*70)

clear_gpu_memory()
model_baseline = MedicalImageClassifier(num_classes=NUM_CLASSES, input_channels=3)
baseline_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

for task_id in range(NUM_TASKS):
    log(f"\n--- Training on Task {task_id + 1}: {WINDOW_NAMES[task_id]} Window ---")
    train_baseline(model_baseline, train_task_datasets[task_id],
                  ITERS_PER_TASK, LEARNING_RATE, BATCH_SIZE, device)

    log(f"\nEvaluating after Task {task_id + 1}:")
    for eval_task_id in range(NUM_TASKS):
        acc = evaluate_accuracy(model_baseline, test_task_datasets[eval_task_id],
                               device, batch_size=EVAL_BATCH_SIZE)
        baseline_matrix[task_id, eval_task_id] = acc
        log(f"  {WINDOW_NAMES[eval_task_id]} Window: {acc:.2f}%")

baseline_metrics = calculate_metrics(baseline_matrix)
log(f"\nBaseline Results:")
log(f"  Average Accuracy: {baseline_metrics['avg_accuracy']:.2f}%")
log(f"  Forgetting: {baseline_metrics['forgetting']:.2f}%")

# EWC EXPERIMENT
log("\n" + "="*70)
log("EWC EXPERIMENT")
log("="*70)

del model_baseline
clear_gpu_memory()
model_ewc = MedicalImageClassifier(num_classes=NUM_CLASSES, input_channels=3)
ewc_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

for task_id in range(NUM_TASKS):
    log(f"\n--- Training on Task {task_id + 1}: {WINDOW_NAMES[task_id]} Window ---")
    train_with_ewc(model_ewc, train_task_datasets[task_id],
                  ITERS_PER_TASK, LEARNING_RATE, BATCH_SIZE, device,
                  current_task=task_id + 1, ewc_lambda=EWC_LAMBDA)

    if task_id < NUM_TASKS - 1:
        log(f"\nEstimating Fisher Information...")
        fisher_info = estimate_fisher_information(model_ewc, train_task_datasets[task_id],
                                                  FISHER_SAMPLES, device)
        model_ewc.fisher_info = fisher_info

        optimal_params = {}
        for name, param in model_ewc.named_parameters():
            if param.requires_grad:
                optimal_params[name.replace('.', '__')] = param.data.clone()
        model_ewc.optimal_params = optimal_params

    log(f"\nEvaluating after Task {task_id + 1}:")
    for eval_task_id in range(NUM_TASKS):
        acc = evaluate_accuracy(model_ewc, test_task_datasets[eval_task_id],
                               device, batch_size=EVAL_BATCH_SIZE)
        ewc_matrix[task_id, eval_task_id] = acc
        log(f"  {WINDOW_NAMES[eval_task_id]} Window: {acc:.2f}%")

ewc_metrics = calculate_metrics(ewc_matrix)
log(f"\nEWC Results:")
log(f"  Average Accuracy: {ewc_metrics['avg_accuracy']:.2f}%")
log(f"  Forgetting: {ewc_metrics['forgetting']:.2f}%")

# REPLAY EXPERIMENT
log("\n" + "="*70)
log("REPLAY EXPERIMENT")
log("="*70)

del model_ewc
clear_gpu_memory()
model_replay = MedicalImageClassifier(num_classes=NUM_CLASSES, input_channels=3)
replay_matrix = np.zeros((NUM_TASKS, NUM_TASKS))
memory_buffer = []

for task_id in range(NUM_TASKS):
    log(f"\n--- Training on Task {task_id + 1}: {WINDOW_NAMES[task_id]} Window ---")

    buffer_dataset = (
        MemorySetDataset(memory_buffer, TARGET_SIZE)
        if len(memory_buffer) > 0 else None
    )

    train_with_replay(model_replay, train_task_datasets[task_id],
                     ITERS_PER_TASK, LEARNING_RATE, BATCH_SIZE, device,
                     current_task=task_id + 1, buffer_dataset=buffer_dataset)

    log(f"\nUpdating memory buffer...")
    new_samples = fill_memory_buffer(train_task_datasets[task_id],
                                     BUFFER_SIZE_PER_TASK, window_type=task_id)
    memory_buffer.extend(new_samples)
    log(f"Buffer size: {len(memory_buffer)} samples")

    log(f"\nEvaluating after Task {task_id + 1}:")
    for eval_task_id in range(NUM_TASKS):
        acc = evaluate_accuracy(model_replay, test_task_datasets[eval_task_id],
                               device, batch_size=EVAL_BATCH_SIZE)
        replay_matrix[task_id, eval_task_id] = acc
        log(f"  {WINDOW_NAMES[eval_task_id]} Window: {acc:.2f}%")

replay_metrics = calculate_metrics(replay_matrix)
log(f"\nReplay Results:")
log(f"  Average Accuracy: {replay_metrics['avg_accuracy']:.2f}%")
log(f"  Forgetting: {replay_metrics['forgetting']:.2f}%")

# EWC + REPLAY EXPERIMENT
log("\n" + "="*70)
log("EWC + REPLAY EXPERIMENT")
log("="*70)

del model_replay
clear_gpu_memory()
model_combined = MedicalImageClassifier(num_classes=NUM_CLASSES, input_channels=3)
combined_matrix = np.zeros((NUM_TASKS, NUM_TASKS))
memory_buffer_combined = []

for task_id in range(NUM_TASKS):
    log(f"\n--- Training on Task {task_id + 1}: {WINDOW_NAMES[task_id]} Window ---")

    buffer_dataset = (
        MemorySetDataset(memory_buffer_combined, TARGET_SIZE)
        if len(memory_buffer_combined) > 0 else None
    )

    train_with_ewc_replay(model_combined, train_task_datasets[task_id],
                         ITERS_PER_TASK, LEARNING_RATE, BATCH_SIZE, device,
                         current_task=task_id + 1, buffer_dataset=buffer_dataset,
                         ewc_lambda=EWC_LAMBDA)

    if task_id < NUM_TASKS - 1:
        log(f"\nEstimating Fisher Information...")
        fisher_info = estimate_fisher_information(model_combined, train_task_datasets[task_id],
                                                  FISHER_SAMPLES, device)
        model_combined.fisher_info = fisher_info

        optimal_params = {}
        for name, param in model_combined.named_parameters():
            if param.requires_grad:
                optimal_params[name.replace('.', '__')] = param.data.clone()
        model_combined.optimal_params = optimal_params

    log(f"\nUpdating memory buffer...")
    new_samples = fill_memory_buffer(train_task_datasets[task_id],
                                     BUFFER_SIZE_PER_TASK, window_type=task_id)
    memory_buffer_combined.extend(new_samples)
    log(f"Buffer size: {len(memory_buffer_combined)} samples")

    log(f"\nEvaluating after Task {task_id + 1}:")
    for eval_task_id in range(NUM_TASKS):
        acc = evaluate_accuracy(model_combined, test_task_datasets[eval_task_id],
                               device, batch_size=EVAL_BATCH_SIZE)
        combined_matrix[task_id, eval_task_id] = acc
        log(f"  {WINDOW_NAMES[eval_task_id]} Window: {acc:.2f}%")

combined_metrics = calculate_metrics(combined_matrix)
log(f"\nCombined Results:")
log(f"  Average Accuracy: {combined_metrics['avg_accuracy']:.2f}%")
log(f"  Forgetting: {combined_metrics['forgetting']:.2f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save plots
if not DEBUG_RUN:
    plot_forgetting_matrix(baseline_matrix, 'Baseline', WINDOW_NAMES,
                          LOG_DIR / f'forgetting_baseline_{timestamp}.png')
    plot_forgetting_matrix(ewc_matrix, 'EWC', WINDOW_NAMES,
                          LOG_DIR / f'forgetting_ewc_{timestamp}.png')
    plot_forgetting_matrix(replay_matrix, 'Replay', WINDOW_NAMES,
                          LOG_DIR / f'forgetting_replay_{timestamp}.png')
    plot_forgetting_matrix(combined_matrix, 'EWC+Replay', WINDOW_NAMES,
                          LOG_DIR / f'forgetting_combined_{timestamp}.png')

# Print summary table
log("\n" + "="*80)
log("FINAL RESULTS SUMMARY - 3-Task CT Window Experiment")
log("="*80)
log(f"{'Method':<20} {'Avg Accuracy':<15} {'Forgetting':<15}")
log("-"*80)

for method_name, metrics in [('Baseline', baseline_metrics), ('EWC', ewc_metrics),
                              ('Replay', replay_metrics), ('EWC+Replay', combined_metrics)]:
    log(f"{method_name:<20} {metrics['avg_accuracy']:>6.2f}%         {metrics['forgetting']:>6.2f}%")

log("="*80)

# Save results to CSV
results_df = pd.DataFrame({
    'Method': ['Baseline', 'EWC', 'Replay', 'EWC+Replay'],
    'Avg_Accuracy': [baseline_metrics['avg_accuracy'], ewc_metrics['avg_accuracy'],
                     replay_metrics['avg_accuracy'], combined_metrics['avg_accuracy']],
    'Forgetting': [baseline_metrics['forgetting'], ewc_metrics['forgetting'],
                   replay_metrics['forgetting'], combined_metrics['forgetting']]
})
results_csv = LOG_DIR / f'results_v2_windows_{timestamp}.csv'
results_df.to_csv(results_csv, index=False)

# Save detailed matrices
np.savez(LOG_DIR / f'matrices_v2_windows_{timestamp}.npz',
         baseline=baseline_matrix, ewc=ewc_matrix,
         replay=replay_matrix, combined=combined_matrix)

# Calculate total execution time
EXPERIMENT_END_TIME = time.time()
total_seconds = EXPERIMENT_END_TIME - EXPERIMENT_START_TIME
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)

log(f"\n{'='*80}")
log(f"EXPERIMENT COMPLETED")
log(f"{'='*80}")
log(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_seconds:.1f} seconds)")
log(f"Results saved to: {results_csv}")
log(f"Matrices saved to: logs/matrices_v2_windows_{timestamp}.npz")
log(f"Full log saved to: {LOG_FILE}")
log(f"{'='*80}")
