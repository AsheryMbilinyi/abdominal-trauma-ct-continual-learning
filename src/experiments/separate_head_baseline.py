# Separate-head baseline experiment.
# Tests whether the shared softmax head is responsible for catastrophic forgetting
# in class-incremental learning by replacing it with per-task output heads.
#
# Architecture:
#   - Shared backbone (conv layers + fc1 + fc2)
#   - Task-specific heads: one Linear(256 -> 2) per task, frozen after its task
#   - Backbone is updated during every task; only heads are task-specific
#
# Evaluation uses a task oracle (task ID known at test time), which is the
# standard evaluation protocol for task-incremental (separate-head) learning.
#
# Two baselines are run for comparison:
#   MultiHead  - separate head per task (proposed baseline)
#   SharedHead - standard shared softmax (same as Stage I exp1)
#
# Run with multiple seeds to report mean +/- std.
#
# Run:
#   python src/experiments/separate_head_baseline.py
#   DEBUG_RUN=1 python src/experiments/separate_head_baseline.py
#   SEEDS=42,123,456 python src/experiments/separate_head_baseline.py

import copy
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
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset

# ── env flags ─────────────────────────────────────────────────────────────────
DEBUG_RUN  = os.environ.get('DEBUG_RUN', '0') == '1'
FORCE_CPU  = os.environ.get('FORCE_CPU', '0') == '1'
FORCE_CUDA = os.environ.get('FORCE_CUDA', '0') == '1'
SEEDS      = list(map(int, os.environ.get('SEEDS', '42,123,456').split(',')))

# ── logging ───────────────────────────────────────────────────────────────────
LOG_DIR   = Path(os.environ.get('LOG_DIR', 'logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE  = LOG_DIR / f'separate_head_{TIMESTAMP}.log'

_fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
_logger = logging.getLogger('sep_head')
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
LABELS_CSV = DATA_DIR / 'train.csv'

MAX_PER_CLASS   = 4   if DEBUG_RUN else 100  # paper: 100 patients/class
MAX_PER_PATIENT = 5   if DEBUG_RUN else 20   # paper: 20 slices/patient
TARGET_SIZE     = (256, 256)
NUM_TASKS       = 2
NUM_CLASSES     = 2
FEAT_DIM        = 256    # output dim of the shared backbone (fc2 output)

ITERS_PER_TASK  = 1   if DEBUG_RUN else 500  # paper CI: 500 iters/task
LR              = 0.001                       # paper CI LR
BATCH_SIZE      = 2   if DEBUG_RUN else 16    # paper CI batch size
EVAL_BS         = 2   if DEBUG_RUN else 64

# ── CT windowing ──────────────────────────────────────────────────────────────
def load_tensor(path, target_size=TARGET_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Cannot read {path}')
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = np.stack([img, img, img], axis=-1)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

# ── datasets ──────────────────────────────────────────────────────────────────
class RSNADataset(Dataset):
    def __init__(self, paths, labels):
        self.image_paths = paths
        self.labels      = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return load_tensor(self.image_paths[idx]), self.labels[idx]


class SubDataset(Dataset):
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

# ── models ────────────────────────────────────────────────────────────────────
class SharedBackbone(nn.Module):
    """Convolutional backbone ending at the penultimate fc layer (output dim=FEAT_DIM)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, FEAT_DIM)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.dropout(F.relu(self.fc2(x)))


class MultiHeadCNN(nn.Module):
    """CNN with one dedicated output head per task.
    During training, backbone is always updated; only the current task's head is trained.
    Previous task heads are frozen to prevent label competition in the output layer.
    """
    def __init__(self, num_tasks, num_classes_per_task=2):
        super().__init__()
        self.backbone = SharedBackbone()
        self.heads    = nn.ModuleList(
            [nn.Linear(FEAT_DIM, num_classes_per_task) for _ in range(num_tasks)]
        )

    def forward(self, x, task_id):
        feat   = self.backbone(x)
        return self.heads[task_id](feat)

    def freeze_head(self, task_id):
        for p in self.heads[task_id].parameters():
            p.requires_grad_(False)

    def unfreeze_head(self, task_id):
        for p in self.heads[task_id].parameters():
            p.requires_grad_(True)


class SharedHeadCNN(nn.Module):
    """Same backbone + single shared output head (the existing Stage-I design)."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = SharedBackbone()
        self.head     = nn.Linear(FEAT_DIM, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

# ── helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if AMP_ENABLED:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _loader(ds, bs, shuffle=True):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=0, drop_last=True,
                      pin_memory=(DEVICE.type == 'cuda'))


def _evaluate_shared(model, dataset, batch_size):
    model.eval().to(DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
                preds = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def _evaluate_multihead(model, dataset, task_id, batch_size):
    """Evaluate multi-head model using the oracle task head."""
    model.eval().to(DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
                preds = model(imgs, task_id=task_id).argmax(1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def calc_metrics(matrix):
    nt  = matrix.shape[0]
    avg = float(matrix[-1].mean())
    fgt = float(np.mean([matrix[t, t] - matrix[-1, t] for t in range(nt - 1)])) if nt > 1 else 0.0
    return {'avg_accuracy': avg, 'forgetting': fgt}

# ── training loops ────────────────────────────────────────────────────────────
def train_multihead_task(model, task_id, dataset, iters, lr, batch_size):
    """Train backbone + head[task_id]; all other heads remain frozen."""
    # Freeze all heads except the current one
    for t in range(len(model.heads)):
        if t != task_id:
            model.freeze_head(t)
        else:
            model.unfreeze_head(t)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    model.train().to(DEVICE)

    loader = _loader(dataset, batch_size)
    it_left = 0
    data_it = None

    for _ in tqdm.tqdm(range(iters), desc=f'MultiHead T{task_id+1}', leave=False):
        if it_left == 0:
            data_it = iter(loader)
            it_left = len(loader)
        imgs, lbls = next(data_it)
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        it_left -= 1

        optimizer.zero_grad()
        with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
            logits = model(imgs, task_id=task_id)
            loss   = criterion(logits, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Freeze the just-trained head to prevent future tasks from overwriting it
    model.freeze_head(task_id)


def train_shared_task(model, dataset, iters, lr, batch_size, task_label):
    """Standard fine-tuning on the shared head (for comparison baseline)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    model.train().to(DEVICE)

    loader  = _loader(dataset, batch_size)
    it_left = 0
    data_it = None

    for _ in tqdm.tqdm(range(iters), desc=f'SharedHead T{task_label}', leave=False):
        if it_left == 0:
            data_it = iter(loader)
            it_left = len(loader)
        imgs, lbls = next(data_it)
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        it_left -= 1

        optimizer.zero_grad()
        with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
            loss = criterion(model(imgs), lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# ── single-seed run ───────────────────────────────────────────────────────────
def run_one_seed(seed, tr_p, tr_l, te_p, te_l):
    log(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    train_base  = RSNADataset(tr_p, tr_l)
    test_base   = RSNADataset(te_p, te_l)
    train_tasks = [SubDataset(train_base, {c}) for c in range(NUM_CLASSES)]
    test_tasks  = [SubDataset(test_base,  {c}) for c in range(NUM_CLASSES)]
    for t, ds in enumerate(train_tasks):
        log(f'  Train task {t}: {len(ds)}  Test task {t}: {len(test_tasks[t])}')

    # ── MultiHead run ────────────────────────────────────────────────────────
    log('\n  [MultiHead - separate task heads, oracle task ID at test time]')
    set_seed(seed)
    mh_model  = MultiHeadCNN(num_tasks=NUM_TASKS).to(DEVICE)
    mh_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

    for t in range(NUM_TASKS):
        train_multihead_task(mh_model, t, train_tasks[t], ITERS_PER_TASK, LR, BATCH_SIZE)
        for j in range(NUM_TASKS):
            acc = _evaluate_multihead(mh_model, test_tasks[j], task_id=j, batch_size=EVAL_BS)
            mh_matrix[t, j] = acc
        log(f'    After T{t+1}: ' + '  '.join(f'T{j+1}={mh_matrix[t,j]:.1f}%' for j in range(NUM_TASKS)))

    mh_metrics = calc_metrics(mh_matrix)
    log(f'    => AvgAcc={mh_metrics["avg_accuracy"]:.2f}%  Forgetting={mh_metrics["forgetting"]:.2f}%')

    # ── SharedHead run ───────────────────────────────────────────────────────
    log('\n  [SharedHead - single shared head (existing Stage-I design)]')
    set_seed(seed)
    sh_model  = SharedHeadCNN(num_classes=NUM_CLASSES).to(DEVICE)
    sh_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

    for t in range(NUM_TASKS):
        train_shared_task(sh_model, train_tasks[t], ITERS_PER_TASK, LR, BATCH_SIZE, task_label=t+1)
        for j in range(NUM_TASKS):
            acc = _evaluate_shared(sh_model, test_tasks[j], batch_size=EVAL_BS)
            sh_matrix[t, j] = acc
        log(f'    After T{t+1}: ' + '  '.join(f'T{j+1}={sh_matrix[t,j]:.1f}%' for j in range(NUM_TASKS)))

    sh_metrics = calc_metrics(sh_matrix)
    log(f'    => AvgAcc={sh_metrics["avg_accuracy"]:.2f}%  Forgetting={sh_metrics["forgetting"]:.2f}%')

    # How much forgetting is eliminated by separate heads?
    fgt_eliminated = sh_metrics['forgetting'] - mh_metrics['forgetting']
    log(f'\n  Head competition accounts for {fgt_eliminated:.2f}% of forgetting '
        f'(SharedHead fgt={sh_metrics["forgetting"]:.2f}%  MultiHead fgt={mh_metrics["forgetting"]:.2f}%)')

    del mh_model, sh_model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'MultiHead':  {'matrix': mh_matrix, 'metrics': mh_metrics},
        'SharedHead': {'matrix': sh_matrix, 'metrics': sh_metrics},
    }

# ── main ──────────────────────────────────────────────────────────────────────
log('=' * 60)
log('Separate-head vs. shared-head baseline')
log(f'Seeds: {SEEDS}')
log(f'Log: {LOG_FILE}')
log('=' * 60)

log('\nLoading data (split fixed at seed=42)...')
tr_p, tr_l, te_p, te_l = load_rsna_data(
    DATA_ROOT, LABELS_CSV,
    max_per_class=MAX_PER_CLASS,
    max_per_patient=MAX_PER_PATIENT,
    seed=42,
)

T0 = time.time()
all_results = {}
for seed in SEEDS:
    all_results[seed] = run_one_seed(seed, tr_p, tr_l, te_p, te_l)

# ── aggregate ─────────────────────────────────────────────────────────────────
log('\n' + '='*70)
log('AGGREGATE RESULTS  (mean +/- std across seeds)')
log('='*70)
log(f'{"Method":<18} {"Avg Acc":>14} {"Forgetting":>14}')
log('-'*70)

rows = []
for method in ('MultiHead', 'SharedHead'):
    avg_accs    = [all_results[s][method]['metrics']['avg_accuracy'] for s in SEEDS]
    forgettings = [all_results[s][method]['metrics']['forgetting']   for s in SEEDS]
    m_acc, s_acc = np.mean(avg_accs),    np.std(avg_accs)
    m_fgt, s_fgt = np.mean(forgettings), np.std(forgettings)
    n = len(SEEDS)
    sem_acc = stats.sem(avg_accs)
    sem_fgt = stats.sem(forgettings)
    ci_acc = (m_acc, m_acc) if sem_acc == 0 else stats.t.interval(0.95, df=n-1, loc=m_acc, scale=sem_acc)
    ci_fgt = (m_fgt, m_fgt) if sem_fgt == 0 else stats.t.interval(0.95, df=n-1, loc=m_fgt, scale=sem_fgt)
    ci_acc = (max(0.0, ci_acc[0]), min(100.0, ci_acc[1]))
    ci_fgt = (max(0.0, ci_fgt[0]), min(100.0, ci_fgt[1]))
    log(f'{method:<18} {m_acc:6.2f} +/- {s_acc:.2f}%  95% CI [{ci_acc[0]:.2f}, {ci_acc[1]:.2f}]'
        f'   Fgt: {m_fgt:.2f} +/- {s_fgt:.2f}%  95% CI [{ci_fgt[0]:.2f}, {ci_fgt[1]:.2f}]')
    for i, seed in enumerate(SEEDS):
        rows.append({'Method': method, 'Seed': seed,
                     'Avg_Accuracy': avg_accs[i], 'Forgetting': forgettings[i],
                     'CI95_Lo': ci_acc[0], 'CI95_Hi': ci_acc[1]})

# Forgetting reduction
mh_fgts = [all_results[s]['MultiHead']['metrics']['forgetting']  for s in SEEDS]
sh_fgts = [all_results[s]['SharedHead']['metrics']['forgetting'] for s in SEEDS]
reduction = np.mean(sh_fgts) - np.mean(mh_fgts)
log(f'\nMean forgetting eliminated by separate heads: {reduction:.2f}%')
log('(positive = separate heads reduce forgetting; '
    'negative = backbone drift dominates)')

out_csv = LOG_DIR / f'separate_head_{TIMESTAMP}.csv'
pd.DataFrame(rows).to_csv(out_csv, index=False)

elapsed = time.time() - T0
log(f'\nDone in {elapsed/60:.1f} min')
log(f'Results CSV : {out_csv}')
log(f'Full log    : {LOG_FILE}')
