# Multi-seed class-incremental experiment.
# Addresses reviewer request for mean +/- std across seeds to verify that
# EWC / Replay / LwF / Baseline conclusions are statistically stable.
# Adds LwF to the Stage-I CNN comparison (not present in exp1_class_incremental.py).
#
# Run:
#   python src/experiments/multi_seed_ci.py
#   DEBUG_RUN=1 python src/experiments/multi_seed_ci.py     # smoke test
#   SEEDS=42,7,99 python src/experiments/multi_seed_ci.py   # custom seeds
#
# Output:
#   logs/multi_seed_ci_per_seed_<ts>.csv   -- per-seed row per method
#   logs/multi_seed_ci_aggregate_<ts>.csv  -- mean +/- std per method

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

# ── env flags ────────────────────────────────────────────────────────────────
DEBUG_RUN  = os.environ.get('DEBUG_RUN', '0') == '1'
FORCE_CPU  = os.environ.get('FORCE_CPU', '0') == '1'
FORCE_CUDA = os.environ.get('FORCE_CUDA', '0') == '1'
SEEDS      = list(map(int, os.environ.get('SEEDS', '42,123,456').split(',')))

# ── logging ──────────────────────────────────────────────────────────────────
LOG_DIR   = Path(os.environ.get('LOG_DIR', 'logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE  = LOG_DIR / f'multi_seed_ci_{TIMESTAMP}.log'

_fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
_logger = logging.getLogger('multi_seed_ci')
_logger.setLevel(logging.INFO)
_logger.addHandler(_fh)

def log(msg):
    print(msg, flush=True)
    _logger.info(msg)

# ── device ───────────────────────────────────────────────────────────────────
if FORCE_CUDA and FORCE_CPU:
    FORCE_CPU = False
if FORCE_CUDA and not torch.cuda.is_available():
    raise RuntimeError('FORCE_CUDA=1 but CUDA is not available.')

DEVICE      = torch.device('cuda' if (FORCE_CUDA or (not FORCE_CPU and torch.cuda.is_available())) else 'cpu')
AMP_ENABLED = DEVICE.type == 'cuda'
AMP_DEVICE  = 'cuda' if DEVICE.type == 'cuda' else 'cpu'
log(f'Device: {DEVICE}  AMP: {AMP_ENABLED}')

# ── data config ──────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.environ.get('DATA_DIR', 'data'))
DATA_ROOT  = DATA_DIR / 'RSNA2023ProcessedImages' / 'soft'
LABELS_CSV = DATA_DIR / 'train.csv'

MAX_PER_CLASS   = 4   if DEBUG_RUN else 100  # paper: 100 patients/class
MAX_PER_PATIENT = 5   if DEBUG_RUN else 20   # paper: 20 slices/patient
TARGET_SIZE     = (256, 256)
NUM_TASKS       = 2
NUM_CLASSES     = 2

ITERS_PER_TASK      = 1   if DEBUG_RUN else 500  # paper CI: 500 iters/task
LR                  = 0.001                       # paper CI LR
BATCH_SIZE          = 2   if DEBUG_RUN else 16    # paper CI batch size
EVAL_BATCH_SIZE     = 2   if DEBUG_RUN else 64
EWC_LAMBDA          = 5000.0                      # paper CI EWC lambda
FISHER_SAMPLES      = 5   if DEBUG_RUN else 200   # paper CI Fisher samples
BUFFER_SIZE         = 5   if DEBUG_RUN else 200   # paper CI replay buffer/task
LWF_ALPHA           = 1.0
LWF_T               = 2.0

METHODS = ['Baseline', 'EWC', 'Replay', 'LwF', 'EWC+Replay']

# ── CT windowing ──────────────────────────────────────────────────────────────
_CT_WINDOWS = {0: (40, 80), 1: (-600, 1500), 2: (50, 400)}

def _apply_window(img, wl, ww):
    lo, hi = wl - ww // 2, wl + ww // 2
    return ((np.clip(img.astype(np.float32), lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)

def load_tensor(path, window_type=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Cannot read {path}')
    if window_type in _CT_WINDOWS:
        img = _apply_window(img, *_CT_WINDOWS[window_type])
    if img.shape[:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img = np.stack([img, img, img], axis=-1)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

# ── datasets ─────────────────────────────────────────────────────────────────
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
    """Filters a base RSNADataset to samples belonging to `sub_labels`."""
    def __init__(self, base, sub_labels):
        self.base    = base
        self.indices = [i for i, l in enumerate(base.labels) if l in sub_labels]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    def get_path_and_label(self, idx):
        oi = self.indices[idx]
        return self.base.image_paths[oi], self.base.labels[oi]


class MemorySet(Dataset):
    """Replay buffer stored as (path, label[, window_type]) tuples."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        path, label = it[0], it[1]
        wt = it[2] if len(it) > 2 else None
        return load_tensor(path, wt), label

# ── data loading ─────────────────────────────────────────────────────────────
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
        log(f'  {split}: {len(paths)} images  class dist: {Counter(labels)}')
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


def calc_metrics(matrix):
    nt = matrix.shape[0]
    avg_acc  = float(matrix[-1].mean())
    fgt_vals = [matrix[t, t] - matrix[-1, t] for t in range(nt - 1)]
    forgetting = float(np.mean(fgt_vals)) if fgt_vals else 0.0
    return {'avg_accuracy': avg_acc, 'forgetting': forgetting}


def estimate_fisher(model, dataset, n_samples):
    fisher = {n.replace('.', '__'): torch.zeros_like(p)
              for n, p in model.named_parameters() if p.requires_grad}
    model.eval().to(DEVICE)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    done = 0
    for imgs, _ in loader:
        if done >= n_samples:
            break
        imgs = imgs.to(DEVICE)
        model.zero_grad()
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)
        for c in range(logits.size(1)):
            model.zero_grad()
            (-torch.log(probs[0, c] + 1e-10)).backward(retain_graph=True)
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n.replace('.', '__')] += probs[0, c].item() * p.grad.pow(2)
        done += 1
    for k in fisher:
        fisher[k] /= max(done, 1)
    return fisher


def make_buffer(task_ds, buf_size, window_type=None):
    """Sample up to buf_size items from a SubDataset for the replay buffer."""
    n    = len(task_ds)
    idxs = random.sample(range(n), min(n, buf_size))
    return [(*task_ds.get_path_and_label(i), window_type) for i in idxs]


def train_task(model, task_loader, replay_loader, optimizer, scaler, criterion,
               iters, fisher=None, star=None, ewc_lambda=0.0,
               teacher=None, lwf_alpha=1.0, lwf_T=2.0, desc='Train'):
    """Unified training loop supporting Baseline / EWC / Replay / LwF / Combined."""
    model.train().to(DEVICE)
    if teacher is not None:
        teacher.eval().to(DEVICE)

    cur_iter = rep_iter = None
    cur_left = rep_left = 0

    for _ in tqdm.tqdm(range(iters), desc=desc, leave=False):
        if cur_left == 0:
            cur_iter = iter(task_loader)
            cur_left = len(task_loader)
        imgs, lbls = next(cur_iter)
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        cur_left -= 1

        if replay_loader is not None:
            if rep_left == 0:
                rep_iter = iter(replay_loader)
                rep_left = len(replay_loader)
            ri, rl = next(rep_iter)
            ri = ri.to(DEVICE, non_blocking=True)
            rl = rl.to(DEVICE, non_blocking=True)
            rep_left -= 1
            imgs = torch.cat([imgs, ri])
            lbls = torch.cat([lbls, rl])

        optimizer.zero_grad()
        with torch.amp.autocast(AMP_DEVICE, enabled=AMP_ENABLED):
            logits = model(imgs)
            loss   = criterion(logits, lbls)

            if fisher is not None and star is not None:
                ewc_pen = sum(
                    (fisher.get(n.replace('.', '__'), torch.zeros_like(p)).to(DEVICE)
                     * (p - star[n.replace('.', '__')].to(DEVICE)).pow(2)).sum()
                    for n, p in model.named_parameters() if p.requires_grad
                )
                loss = loss + (ewc_lambda / 2) * ewc_pen

            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                kd = -(F.softmax(t_logits / lwf_T, dim=1) *
                        F.log_softmax(logits / lwf_T, dim=1)).sum(dim=1).mean()
                loss = loss + lwf_alpha * (lwf_T ** 2) * kd

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# ── single-seed run ───────────────────────────────────────────────────────────
def run_one_seed(seed, tr_p, tr_l, te_p, te_l):
    log(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    # Build datasets (same data paths, fresh Dataset objects)
    train_base = RSNADataset(tr_p, tr_l)
    test_base  = RSNADataset(te_p, te_l)
    train_tasks = [SubDataset(train_base, {c}) for c in range(NUM_CLASSES)]
    test_tasks  = [SubDataset(test_base, {c})  for c in range(NUM_CLASSES)]

    results = {}

    for method in METHODS:
        log(f'\n  [{method}]')
        # Reset seed so every method starts from identical weights for this seed
        set_seed(seed)

        model     = CNN(num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        scaler    = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
        matrix    = np.zeros((NUM_TASKS, NUM_TASKS))

        memory  = []    # replay buffer: list of (path, label, window_type)
        fisher  = None  # EWC Fisher dict
        star    = None  # EWC star params
        teacher = None  # LwF frozen teacher

        for t in range(NUM_TASKS):
            task_loader   = _loader(train_tasks[t], BATCH_SIZE)
            replay_loader = (_loader(MemorySet(memory), BATCH_SIZE)
                             if memory else None)

            use_ewc    = method in ('EWC', 'EWC+Replay') and t > 0
            use_replay = method in ('Replay', 'EWC+Replay') and bool(memory)
            use_lwf    = method == 'LwF' and teacher is not None

            train_task(
                model, task_loader,
                replay_loader if use_replay else None,
                optimizer, scaler, criterion,
                ITERS_PER_TASK,
                fisher=fisher if use_ewc else None,
                star=star    if use_ewc else None,
                ewc_lambda=EWC_LAMBDA if use_ewc else 0.0,
                teacher=teacher if use_lwf else None,
                lwf_alpha=LWF_ALPHA,
                lwf_T=LWF_T,
                desc=f'{method} T{t+1}',
            )

            # Update CL state
            if method in ('EWC', 'EWC+Replay'):
                fisher = estimate_fisher(model, train_tasks[t], FISHER_SAMPLES)
                star   = {n.replace('.', '__'): p.detach().clone()
                          for n, p in model.named_parameters() if p.requires_grad}

            if method == 'LwF':
                teacher = copy.deepcopy(model).eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)

            if method in ('Replay', 'EWC+Replay'):
                memory.extend(make_buffer(train_tasks[t], BUFFER_SIZE))

            # Evaluate on all tasks after training task t
            for j in range(NUM_TASKS):
                acc = evaluate(model, test_tasks[j], EVAL_BATCH_SIZE)
                matrix[t, j] = acc
            log(f'    After T{t+1}: ' + '  '.join(f'T{j+1}={matrix[t,j]:.1f}%'
                                                    for j in range(NUM_TASKS)))

        metrics = calc_metrics(matrix)
        log(f'    => AvgAcc={metrics["avg_accuracy"]:.2f}%  Forgetting={metrics["forgetting"]:.2f}%')
        results[method] = {'matrix': matrix, 'metrics': metrics}

        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    return results

# ── main ─────────────────────────────────────────────────────────────────────
log('=' * 60)
log('Multi-seed class-incremental experiment')
log(f'Seeds: {SEEDS}')
log(f'Methods: {METHODS}')
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
log('\n' + '=' * 70)
log('AGGREGATE RESULTS  (mean +/- std across seeds)')
log('=' * 70)
log(f'{"Method":<18} {"Avg Acc":>14} {"Forgetting":>14}  Per-task final acc (mean)')
log('-' * 80)

per_seed_rows = []
agg_rows      = []

for method in METHODS:
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

    # Per-task final accuracy: shows class-specific collapse e.g. [0%, 100%]
    task_means = []
    for t in range(NUM_TASKS):
        task_accs = [all_results[s][method]['matrix'][-1, t] for s in SEEDS]
        task_means.append(np.mean(task_accs))
    task_str = '  '.join(f'T{t+1}={task_means[t]:.1f}%' for t in range(NUM_TASKS))

    log(f'{method:<18} {m_acc:6.2f} +/- {s_acc:.2f}%  95% CI [{ci_acc[0]:.2f}, {ci_acc[1]:.2f}]'
        f'   Fgt: {m_fgt:.2f} +/- {s_fgt:.2f}%  95% CI [{ci_fgt[0]:.2f}, {ci_fgt[1]:.2f}]'
        f'   [{task_str}]')

    for i, seed in enumerate(SEEDS):
        row = {'Method': method, 'Seed': seed,
               'Avg_Accuracy': avg_accs[i], 'Forgetting': forgettings[i]}
        for t in range(NUM_TASKS):
            row[f'Task{t+1}_Final_Acc'] = all_results[seed][method]['matrix'][-1, t]
        per_seed_rows.append(row)

    agg_row = {
        'Method': method,
        'Avg_Acc_Mean': m_acc, 'Avg_Acc_Std': s_acc,
        'Avg_Acc_CI95_Lo': ci_acc[0], 'Avg_Acc_CI95_Hi': ci_acc[1],
        'Forgetting_Mean': m_fgt, 'Forgetting_Std': s_fgt,
        'Forgetting_CI95_Lo': ci_fgt[0], 'Forgetting_CI95_Hi': ci_fgt[1],
        'Num_Seeds': len(SEEDS),
    }
    for t in range(NUM_TASKS):
        task_accs = [all_results[s][method]['matrix'][-1, t] for s in SEEDS]
        agg_row[f'Task{t+1}_Acc_Mean'] = np.mean(task_accs)
        agg_row[f'Task{t+1}_Acc_Std']  = np.std(task_accs)
    agg_rows.append(agg_row)

per_seed_csv = LOG_DIR / f'multi_seed_ci_per_seed_{TIMESTAMP}.csv'
agg_csv      = LOG_DIR / f'multi_seed_ci_aggregate_{TIMESTAMP}.csv'
pd.DataFrame(per_seed_rows).to_csv(per_seed_csv, index=False)
pd.DataFrame(agg_rows).to_csv(agg_csv, index=False)

elapsed = time.time() - T0
log(f'\nDone in {elapsed/60:.1f} min')
log(f'Per-seed CSV  : {per_seed_csv}')
log(f'Aggregate CSV : {agg_csv}')
log(f'Full log      : {LOG_FILE}')
