# Joint-training upper bound experiment.
# Trains a single model on ALL tasks simultaneously (no continual learning).
# This is the oracle ceiling that continual learning methods are compared against.
# Addresses reviewer request for a joint-training reference to quantify the
# performance gap introduced by sequential training.
#
# Two settings are evaluated:
#   CI  – class-incremental: classes 0+1 combined (Stage I, Exp 1 analogue)
#   DI  – domain-incremental: brain+lung+soft-tissue windows combined (Stage I, Exp 2 analogue)
#
# Run:
#   python src/experiments/joint_training_upper_bound.py
#   DEBUG_RUN=1 python src/experiments/joint_training_upper_bound.py

import logging
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# ── env flags ─────────────────────────────────────────────────────────────────
DEBUG_RUN  = os.environ.get('DEBUG_RUN', '0') == '1'
FORCE_CPU  = os.environ.get('FORCE_CPU', '0') == '1'
FORCE_CUDA = os.environ.get('FORCE_CUDA', '0') == '1'

# ── logging ───────────────────────────────────────────────────────────────────
LOG_DIR   = Path(os.environ.get('LOG_DIR', 'logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE  = LOG_DIR / f'joint_training_upper_bound_{TIMESTAMP}.log'

_fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
_logger = logging.getLogger('joint_ub')
_logger.setLevel(logging.INFO)
_logger.addHandler(_fh)

def log(msg):
    print(msg, flush=True)
    _logger.info(msg)

# ── device ────────────────────────────────────────────────────────────────────
if FORCE_CUDA and FORCE_CPU:
    FORCE_CPU = False
if FORCE_CUDA and not torch.cuda.is_available():
    raise RuntimeError('FORCE_CUDA=1 but CUDA not available.')

DEVICE      = torch.device('cuda' if (FORCE_CUDA or (not FORCE_CPU and torch.cuda.is_available())) else 'cpu')
AMP_ENABLED = DEVICE.type == 'cuda'
AMP_DEVICE  = 'cuda' if DEVICE.type == 'cuda' else 'cpu'
log(f'Device: {DEVICE}  AMP: {AMP_ENABLED}')

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.environ.get('DATA_DIR', 'data'))
DATA_ROOT  = DATA_DIR / 'RSNA2023ProcessedImages' / 'soft'

# Pre-windowed directories for the domain-incremental experiment
DI_ROOTS = {
    0: DATA_DIR / 'RSNA2023ProcessedImages' / 'brain',
    1: DATA_DIR / 'RSNA2023ProcessedImages' / 'lung',
    2: DATA_DIR / 'RSNA2023ProcessedImages' / 'soft',
}
LABELS_CSV = DATA_DIR / 'train.csv'

MAX_PER_CLASS   = 4  if DEBUG_RUN else 100  # paper: 100 patients/class
MAX_PER_PATIENT = 5  if DEBUG_RUN else 20   # paper: 20 slices/patient
TARGET_SIZE     = (256, 256)
NUM_CI_TASKS    = 2
NUM_DI_TASKS    = 3

EPOCHS      = 1   if DEBUG_RUN else 5    # 5 epochs enough for joint training   # joint training converges faster than sequential
LR          = 0.005
BATCH_SIZE  = 2  if DEBUG_RUN else 32
EVAL_BS     = 2  if DEBUG_RUN else 64
SEED        = 42

# CT windows used by the domain-incremental experiment (one window per task)
DI_WINDOWS = {0: ('Brain',       40,  80),
              1: ('Lung',       -600, 1500),
              2: ('SoftTissue',   50, 400)}

# ── CT windowing ──────────────────────────────────────────────────────────────
def _apply_window(img, wl, ww):
    lo, hi = wl - ww // 2, wl + ww // 2
    return ((np.clip(img.astype(np.float32), lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)

def load_tensor(path, window_type=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Cannot read {path}')
    if window_type in DI_WINDOWS:
        _, wl, ww = DI_WINDOWS[window_type]
        img = _apply_window(img, wl, ww)
    if img.shape[:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img = np.stack([img, img, img], axis=-1)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

# ── datasets ──────────────────────────────────────────────────────────────────
class RSNADataset(Dataset):
    def __init__(self, paths, labels, window_type=None):
        self.image_paths = paths
        self.labels      = labels
        self.window_type = window_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return load_tensor(self.image_paths[idx], self.window_type), self.labels[idx]


class SubDataset(Dataset):
    """Filters to samples belonging to the given class labels."""
    def __init__(self, base, sub_labels):
        self.base    = base
        self.indices = [i for i, l in enumerate(base.labels) if l in sub_labels]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

# ── data loading ──────────────────────────────────────────────────────────────
def load_rsna_data(data_root, labels_csv, test_split=0.2,
                   max_per_class=None, max_per_patient=None, seed=42):
    df = pd.read_csv(labels_csv)
    pid_label = {int(r.patient_id): int(r.any_injury)
                 for _, r in df[['patient_id', 'any_injury']].dropna().iterrows()}

    by_label = {}
    for pid, lbl in pid_label.items():
        by_label.setdefault(lbl, []).append(pid)

    rng = random.Random(seed)
    train_pids, test_pids = [], []
    for lbl, pids in sorted(by_label.items()):
        rng.shuffle(pids)
        if max_per_class:
            pids = pids[:max_per_class]
        cut = int(len(pids) * (1 - test_split))
        train_pids.extend(pids[:cut])
        test_pids.extend(pids[cut:])

    def collect(pids, split):
        paths, labels = [], []
        for pid in pids:
            d = Path(data_root) / str(pid)
            if not d.exists():
                continue
            imgs = []
            for sd in d.iterdir():
                if not sd.is_dir():
                    continue
                for p in sd.glob('*.png'):
                    imgs.append(p)
                    if max_per_patient and len(imgs) >= max_per_patient:
                        break
                if max_per_patient and len(imgs) >= max_per_patient:
                    break
            lbl = pid_label.get(pid)
            if lbl is None or not imgs:
                continue
            paths.extend([str(p) for p in imgs])
            labels.extend([lbl] * len(imgs))
        log(f'  {split}: {len(paths)} images  dist: {Counter(labels)}')
        return paths, labels

    tr_p, tr_l = collect(train_pids, 'train')
    te_p, te_l = collect(test_pids, 'test')
    return tr_p, tr_l, te_p, te_l

# ── model ─────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

# ── training / evaluation ─────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if AMP_ENABLED:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_joint(model, joint_dataset, epochs, lr, batch_size):
    """Standard epoch-based training on the joint (all-task) dataset."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    model.train().to(DEVICE)

    loader = DataLoader(joint_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True,
                        pin_memory=(DEVICE.type == 'cuda'))
    log(f'  Joint dataset size: {len(joint_dataset)}  epochs: {epochs}')

    for epoch in tqdm.tqdm(range(1, epochs + 1), desc='Joint training'):
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


def evaluate(model, dataset, batch_size):
    model.eval().to(DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
                preds = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

# ── Experiment 1: Class-Incremental Joint Training ────────────────────────────
def run_ci_joint(tr_p, tr_l, te_p, te_l):
    log('\n' + '='*60)
    log('EXPERIMENT 1: Class-Incremental Joint Training (upper bound)')
    log('='*60)
    log('Train on class 0 + class 1 simultaneously, evaluate per class.')

    set_seed(SEED)

    # Full dataset contains both classes; task test sets are class-stratified
    train_full = RSNADataset(tr_p, tr_l)
    test_full  = RSNADataset(te_p, te_l)

    task_test = [SubDataset(test_full, {c}) for c in range(NUM_CI_TASKS)]
    for t, ds in enumerate(task_test):
        log(f'  Test task {t}: {len(ds)} samples (class {t})')

    model = CNN(num_classes=2).to(DEVICE)
    train_joint(model, train_full, EPOCHS, LR, BATCH_SIZE)

    results = {}
    log('\n  Task accuracies after joint training:')
    for t in range(NUM_CI_TASKS):
        acc = evaluate(model, task_test[t], EVAL_BS)
        results[f'Task{t+1}_Class{t}'] = acc
        log(f'    Task {t+1} (class {t}): {acc:.2f}%')

    # Also report full-dataset accuracy for reference
    full_acc = evaluate(model, test_full, EVAL_BS)
    results['Overall'] = full_acc
    log(f'    Overall (all classes): {full_acc:.2f}%')

    return results


# ── Experiment 2: Domain-Incremental Joint Training ───────────────────────────
def run_di_joint(tr_p, tr_l, te_p, te_l):
    log('\n' + '='*60)
    log('EXPERIMENT 2: Domain-Incremental Joint Training (upper bound)')
    log('='*60)
    log('Train on Brain+Lung+SoftTissue windows simultaneously, evaluate per window.')

    set_seed(SEED)

    # Load each window task from its pre-windowed PNG directory (same patients, different appearance)
    di_train_tasks, di_test_tasks = [], []
    for t, (name, _, _) in DI_WINDOWS.items():
        t_tr_p, t_tr_l, t_te_p, t_te_l = load_rsna_data(
            DI_ROOTS[t], LABELS_CSV,
            max_per_class=MAX_PER_CLASS,
            max_per_patient=MAX_PER_PATIENT,
            seed=SEED,
        )
        di_train_tasks.append(RSNADataset(t_tr_p, t_tr_l))
        di_test_tasks.append(RSNADataset(t_te_p, t_te_l))
        log(f'  Task {t+1} ({name}): {len(t_tr_p)} train, {len(t_te_p)} test')

    train_tasks, test_tasks = di_train_tasks, di_test_tasks

    # Joint dataset = all three windows concatenated
    joint_train = ConcatDataset(train_tasks)
    log(f'\n  Joint training dataset size: {len(joint_train)} (3x all patients)')

    model = CNN(num_classes=2).to(DEVICE)
    train_joint(model, joint_train, EPOCHS, LR, BATCH_SIZE)

    results = {}
    log('\n  Task accuracies after joint training:')
    for t, (name, _, _) in DI_WINDOWS.items():
        acc = evaluate(model, test_tasks[t], EVAL_BS)
        results[f'Task{t+1}_{name}'] = acc
        log(f'    Task {t+1} ({name} window): {acc:.2f}%')

    return results


# ── main ──────────────────────────────────────────────────────────────────────
log('=' * 60)
log('Joint-Training Upper Bound')
log(f'Log: {LOG_FILE}')
log('=' * 60)

log('\nLoading data...')
tr_p, tr_l, te_p, te_l = load_rsna_data(
    DATA_ROOT, LABELS_CSV,
    max_per_class=MAX_PER_CLASS,
    max_per_patient=MAX_PER_PATIENT,
    seed=SEED,
)

T0 = time.time()

ci_results = run_ci_joint(tr_p, tr_l, te_p, te_l)
di_results = run_di_joint(tr_p, tr_l, te_p, te_l)

# ── summary ───────────────────────────────────────────────────────────────────
log('\n' + '='*60)
log('SUMMARY: Joint-training upper bounds')
log('='*60)
log('\nClass-Incremental (CI):')
for k, v in ci_results.items():
    log(f'  {k}: {v:.2f}%')

log('\nDomain-Incremental (DI):')
for k, v in di_results.items():
    log(f'  {k}: {v:.2f}%')

rows = []
for k, v in ci_results.items():
    rows.append({'Setting': 'CI', 'Task': k, 'JointAcc': v})
for k, v in di_results.items():
    rows.append({'Setting': 'DI', 'Task': k, 'JointAcc': v})

out_csv = LOG_DIR / f'joint_training_upper_bound_{TIMESTAMP}.csv'
pd.DataFrame(rows).to_csv(out_csv, index=False)

elapsed = time.time() - T0
log(f'\nDone in {elapsed/60:.1f} min')
log(f'Results CSV : {out_csv}')
log(f'Full log    : {LOG_FILE}')
