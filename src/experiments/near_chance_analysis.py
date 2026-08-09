# Near-chance diagnostic experiment.
# Addresses reviewer request to distinguish catastrophic forgetting from
# failure to learn the task in the first place, especially when accuracy
# stays near chance level (~50% for binary classification).
#
# For every (method, task) pair where the FINAL accuracy <= CHANCE_THRESHOLD:
#   1. Check the diagonal (accuracy immediately after training that task).
#      - If diagonal is good but final is near chance => FORGETTING
#      - If diagonal is also near chance => NEVER LEARNED under CL setup
#   2. Train a fresh model on that task alone (no CL, single task).
#      - If fresh model fails too => TASK UNLEARNABLE with this setup
#      - If fresh model succeeds but CL fails => CL-specific failure
#   3. Compare to majority-class baseline (trivial predictor).
#
# Verdict logic:
#   diagonal > threshold AND final <= threshold                  -> Forgetting
#   diagonal <= threshold AND fresh_acc > threshold             -> CL Failure (not forgetting; task is learnable solo)
#   diagonal <= threshold AND fresh_acc <= threshold            -> Task Unlearnable (setup-level issue)
#
# The script runs a compact version of all five methods to obtain accuracy
# matrices efficiently, then runs fresh solo models for near-chance entries.
#
# Run:
#   python src/experiments/near_chance_analysis.py
#   DEBUG_RUN=1 python src/experiments/near_chance_analysis.py

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
from torch.utils.data import DataLoader, Dataset

# ── env flags ─────────────────────────────────────────────────────────────────
DEBUG_RUN  = os.environ.get('DEBUG_RUN', '0') == '1'
FORCE_CPU  = os.environ.get('FORCE_CPU', '0') == '1'
FORCE_CUDA = os.environ.get('FORCE_CUDA', '0') == '1'

CHANCE_THRESHOLD = float(os.environ.get('CHANCE_THRESHOLD', '55.0'))

# ── logging ───────────────────────────────────────────────────────────────────
LOG_DIR   = Path(os.environ.get('LOG_DIR', 'logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE  = LOG_DIR / f'near_chance_analysis_{TIMESTAMP}.log'

_fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
_logger = logging.getLogger('near_chance')
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
log(f'Device: {DEVICE}  AMP: {AMP_ENABLED}  Chance threshold: {CHANCE_THRESHOLD}%')

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.environ.get('DATA_DIR', 'data'))
DATA_ROOT  = DATA_DIR / 'RSNA2023ProcessedImages' / 'soft'

# Pre-windowed directories for the domain-incremental experiment
_DI_ROOTS = {
    0: DATA_DIR / 'RSNA2023ProcessedImages' / 'brain',
    1: DATA_DIR / 'RSNA2023ProcessedImages' / 'lung',
    2: DATA_DIR / 'RSNA2023ProcessedImages' / 'soft',
}
LABELS_CSV = DATA_DIR / 'train.csv'

MAX_PER_CLASS   = 4   if DEBUG_RUN else 100  # paper: 100 patients/class
MAX_PER_PATIENT = 5   if DEBUG_RUN else 20   # paper: 20 slices/patient
TARGET_SIZE     = (256, 256)
NUM_TASKS       = 2
NUM_CLASSES     = 2
SEED            = 42

ITERS_PER_TASK = 1   if DEBUG_RUN else 500  # paper CI: 500 iters/task
FRESH_ITERS    = 1   if DEBUG_RUN else 500
LR             = 0.001                       # paper CI LR
BATCH_SIZE     = 2   if DEBUG_RUN else 16    # paper CI batch size
EVAL_BS        = 2   if DEBUG_RUN else 64
EWC_LAMBDA     = 5000.0                      # paper CI EWC lambda
FISHER_SAMPLES = 5   if DEBUG_RUN else 200   # paper CI Fisher samples
BUFFER_SIZE    = 5   if DEBUG_RUN else 200   # paper CI replay buffer/task
LWF_ALPHA      = 1.0
LWF_T          = 2.0

METHODS = ['Baseline', 'EWC', 'Replay', 'LwF', 'EWC+Replay']

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

    def get_path_and_label(self, idx):
        oi = self.indices[idx]
        return self.base.image_paths[oi], self.base.labels[oi]


class MemorySet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx][0], self.items[idx][1]
        return load_tensor(path), label

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
        return paths, labels

    tr_p, tr_l = collect(train_pids, 'train')
    te_p, te_l = collect(test_pids, 'test')
    log(f'Train: {len(tr_p)} images  Test: {len(te_p)} images  dist: {Counter(tr_l)}')
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


def evaluate(model, dataset, batch_size=64):
    if len(dataset) == 0:
        return 0.0
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


def majority_baseline(dataset):
    """Accuracy of always predicting the most frequent class."""
    lbls = [dataset.base.labels[dataset.indices[i]] for i in range(len(dataset))]
    cnt  = Counter(lbls)
    return 100.0 * cnt.most_common(1)[0][1] / len(lbls) if lbls else 50.0


def make_buffer(task_ds, buf_size):
    n    = len(task_ds)
    idxs = random.sample(range(n), min(n, buf_size))
    return [(*task_ds.get_path_and_label(i), None) for i in idxs]


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


def _loader(ds, bs):
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                      drop_last=True, pin_memory=(DEVICE.type == 'cuda'))


def train_loop(model, task_loader, replay_loader, optimizer, scaler, criterion,
               iters, fisher=None, star=None, ewc_lambda=0.0,
               teacher=None, lwf_alpha=1.0, lwf_T=2.0, desc='Train'):
    import copy as _copy
    model.train().to(DEVICE)
    if teacher is not None:
        teacher.eval().to(DEVICE)

    cur_it = rep_it = None
    cur_left = rep_left = 0

    for _ in tqdm.tqdm(range(iters), desc=desc, leave=False):
        if cur_left == 0:
            cur_it = iter(task_loader)
            cur_left = len(task_loader)
        imgs, lbls = next(cur_it)
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        cur_left -= 1

        if replay_loader is not None:
            if rep_left == 0:
                rep_it = iter(replay_loader)
                rep_left = len(replay_loader)
            ri, rl = next(rep_it)
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

# ── run all CL methods and collect matrices ────────────────────────────────────
def run_all_methods(train_tasks, test_tasks):
    """Run all CL methods and return accuracy matrices."""
    import copy as _copy
    matrices = {}

    for method in METHODS:
        log(f'  [{method}]')
        set_seed(SEED)
        model     = CNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        scaler    = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
        matrix    = np.zeros((NUM_TASKS, NUM_TASKS))

        memory  = []
        fisher  = None
        star    = None
        teacher = None

        for t in range(NUM_TASKS):
            task_ldr  = _loader(train_tasks[t], BATCH_SIZE)
            rep_ldr   = _loader(MemorySet(memory), BATCH_SIZE) if memory else None
            use_ewc   = method in ('EWC', 'EWC+Replay') and t > 0
            use_rep   = method in ('Replay', 'EWC+Replay') and bool(memory)
            use_lwf   = method == 'LwF' and teacher is not None

            train_loop(
                model, task_ldr,
                rep_ldr if use_rep else None,
                optimizer, scaler, criterion,
                ITERS_PER_TASK,
                fisher=fisher if use_ewc else None,
                star=star    if use_ewc else None,
                ewc_lambda=EWC_LAMBDA if use_ewc else 0.0,
                teacher=teacher if use_lwf else None,
                lwf_alpha=LWF_ALPHA, lwf_T=LWF_T,
                desc=f'{method} T{t+1}',
            )

            if method in ('EWC', 'EWC+Replay'):
                fisher = estimate_fisher(model, train_tasks[t], FISHER_SAMPLES)
                star   = {n.replace('.', '__'): p.detach().clone()
                          for n, p in model.named_parameters() if p.requires_grad}

            if method == 'LwF':
                teacher = _copy.deepcopy(model).eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)

            if method in ('Replay', 'EWC+Replay'):
                memory.extend(make_buffer(train_tasks[t], BUFFER_SIZE))

            for j in range(NUM_TASKS):
                matrix[t, j] = evaluate(model, test_tasks[j], EVAL_BS)

        matrices[method] = matrix
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    return matrices


def run_fresh_solo(train_task_ds, test_task_ds, task_id):
    """Train a fresh model on a single task only; returns test accuracy."""
    log(f'    Fresh solo model for task {task_id+1}...')
    set_seed(SEED)
    model     = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)

    loader = _loader(train_task_ds, BATCH_SIZE)
    it_left = 0
    data_it = None
    model.train()

    for _ in tqdm.tqdm(range(FRESH_ITERS), desc=f'Fresh T{task_id+1}', leave=False):
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

    acc = evaluate(model, test_task_ds, EVAL_BS)
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return acc


# ── Domain-incremental helpers ────────────────────────────────────────────────
# The paper's most prominent near-chance finding is in the DI experiment where
# ALL methods stay ~50% even immediately after training each window task.
# Adding DI analysis here answers: is it forgetting, CL failure, or the
# label noise/representation bottleneck the paper identifies as the root cause?

_CT_WINDOWS_DI = {
    0: ('Brain',       40,   80),
    1: ('Lung',       -600, 1500),
    2: ('SoftTissue',  50,  400),
}
NUM_DI_TASKS = 3


class DIMemorySet(Dataset):
    """Replay buffer for DI: stores (path, label) tuples from pre-windowed dirs."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx][0], self.items[idx][1]
        return load_tensor(path), label


def make_di_buffer(ds, buf_size):
    n    = len(ds)
    idxs = random.sample(range(n), min(n, buf_size))
    return [(ds.image_paths[i], ds.labels[i], None) for i in idxs]


def majority_di(ds):
    cnt = Counter(ds.labels)
    return 100.0 * cnt.most_common(1)[0][1] / len(ds.labels)


def create_di_tasks():
    """Load DI tasks from pre-windowed PNG directories (brain/lung/soft subdirs)."""
    train_tasks, test_tasks = [], []
    for t in range(NUM_DI_TASKS):
        t_tr_p, t_tr_l, t_te_p, t_te_l = load_rsna_data(
            _DI_ROOTS[t], LABELS_CSV,
            max_per_class=MAX_PER_CLASS,
            max_per_patient=MAX_PER_PATIENT,
            seed=SEED,
        )
        train_tasks.append(RSNADataset(t_tr_p, t_tr_l))
        test_tasks.append(RSNADataset(t_te_p, t_te_l))
    return train_tasks, test_tasks


def run_all_methods_di(train_tasks, test_tasks):
    """Run all CL methods on the 3-task domain-incremental stream."""
    import copy as _copy
    matrices = {}

    for method in METHODS:
        log(f'  [DI {method}]')
        set_seed(SEED)
        model     = CNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        scaler    = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
        matrix    = np.zeros((NUM_DI_TASKS, NUM_DI_TASKS))

        memory  = []
        fisher  = None
        star    = None
        teacher = None

        for t in range(NUM_DI_TASKS):
            task_ldr = _loader(train_tasks[t], BATCH_SIZE)
            rep_ldr  = _loader(DIMemorySet(memory), BATCH_SIZE) if memory else None
            use_ewc  = method in ('EWC', 'EWC+Replay') and t > 0
            use_rep  = method in ('Replay', 'EWC+Replay') and bool(memory)
            use_lwf  = method == 'LwF' and teacher is not None

            train_loop(
                model, task_ldr,
                rep_ldr if use_rep else None,
                optimizer, scaler, criterion,
                ITERS_PER_TASK,
                fisher=fisher if use_ewc else None,
                star=star    if use_ewc else None,
                ewc_lambda=EWC_LAMBDA if use_ewc else 0.0,
                teacher=teacher if use_lwf else None,
                lwf_alpha=LWF_ALPHA, lwf_T=LWF_T,
                desc=f'DI {method} T{t+1}',
            )

            if method in ('EWC', 'EWC+Replay'):
                fisher = estimate_fisher(model, train_tasks[t], FISHER_SAMPLES)
                star   = {n.replace('.', '__'): p.detach().clone()
                          for n, p in model.named_parameters() if p.requires_grad}

            if method == 'LwF':
                teacher = _copy.deepcopy(model).eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)

            if method in ('Replay', 'EWC+Replay'):
                memory.extend(make_di_buffer(train_tasks[t], BUFFER_SIZE))

            for j in range(NUM_DI_TASKS):
                matrix[t, j] = evaluate(model, test_tasks[j], EVAL_BS)

        matrices[method] = matrix
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    return matrices


def assign_verdict(diagonal_acc, final_acc, fresh_acc, threshold):
    if final_acc > threshold:
        return 'OK'
    if diagonal_acc > threshold:
        return 'Forgetting'        # learned OK, then forgotten
    if fresh_acc > threshold:
        return 'CL Failure'        # task is learnable solo, but CL setup prevents it
    return 'Task Unlearnable'      # fails even solo -> architecture/data issue


# ── main ──────────────────────────────────────────────────────────────────────
log('=' * 60)
log('Near-chance diagnostic analysis')
log(f'Chance threshold: {CHANCE_THRESHOLD}%')
log(f'Log: {LOG_FILE}')
log('=' * 60)

log('\nLoading data...')
tr_p, tr_l, te_p, te_l = load_rsna_data(
    DATA_ROOT, LABELS_CSV,
    max_per_class=MAX_PER_CLASS,
    max_per_patient=MAX_PER_PATIENT,
    seed=SEED,
)

train_base  = RSNADataset(tr_p, tr_l)
test_base   = RSNADataset(te_p, te_l)
train_tasks = [SubDataset(train_base, {c}) for c in range(NUM_CLASSES)]
test_tasks  = [SubDataset(test_base,  {c}) for c in range(NUM_CLASSES)]

# Majority baselines (one per task)
majority_accs = [majority_baseline(test_tasks[t]) for t in range(NUM_TASKS)]
log(f'Majority-class baselines: ' + '  '.join(f'T{t+1}={majority_accs[t]:.1f}%' for t in range(NUM_TASKS)))

T0 = time.time()
log('\nRunning all CL methods...')
matrices = run_all_methods(train_tasks, test_tasks)

# ── identify near-chance entries and run fresh models ─────────────────────────
log(f'\nScanning for near-chance entries (final acc <= {CHANCE_THRESHOLD}%)...')

fresh_cache = {}   # task_id -> fresh_acc (run once per task)
rows = []

for method in METHODS:
    matrix = matrices[method]
    for t in range(NUM_TASKS):
        diagonal_acc = float(matrix[t, t])        # accuracy right after training task t
        final_acc    = float(matrix[-1, t])        # accuracy after all tasks
        maj_acc      = majority_accs[t]

        if final_acc <= CHANCE_THRESHOLD:
            if t not in fresh_cache:
                fresh_cache[t] = run_fresh_solo(train_tasks[t], test_tasks[t], task_id=t)
            fresh_acc = fresh_cache[t]
        else:
            fresh_acc = None   # not needed for above-threshold entries

        verdict = assign_verdict(
            diagonal_acc, final_acc,
            fresh_cache.get(t, CHANCE_THRESHOLD + 1),
            CHANCE_THRESHOLD,
        )

        rows.append({
            'Setting':      'CI',
            'Method':       method,
            'Task':         t + 1,
            'Majority_Acc': round(maj_acc, 2),
            'Diagonal_Acc': round(diagonal_acc, 2),
            'Final_Acc':    round(final_acc, 2),
            'Fresh_Acc':    round(fresh_acc, 2) if fresh_acc is not None else 'N/A',
            'Near_Chance':  final_acc <= CHANCE_THRESHOLD,
            'Verdict':      verdict,
        })

# ── Domain-incremental near-chance analysis ───────────────────────────────────
# The paper found ALL methods stay ~50% in DI. This block checks:
#   - Is the diagonal (just-trained) also near chance? (never learned)
#   - Does a fresh solo model on one window task succeed? (learnable solo?)
# If fresh solo also fails -> Task Unlearnable -> label noise is the bottleneck.
log('\n' + '='*60)
log('DOMAIN-INCREMENTAL near-chance analysis (3 window tasks)')
log('='*60)

di_train_tasks, di_test_tasks = create_di_tasks()
di_majority = [majority_di(di_test_tasks[t]) for t in range(NUM_DI_TASKS)]
log('Majority-class baselines: ' + '  '.join(
    f'{_CT_WINDOWS_DI[t][0]}={di_majority[t]:.1f}%' for t in range(NUM_DI_TASKS)))

log('\nRunning all CL methods on DI stream...')
di_matrices = run_all_methods_di(di_train_tasks, di_test_tasks)

di_fresh_cache = {}

for method in METHODS:
    matrix = di_matrices[method]
    for t in range(NUM_DI_TASKS):
        diagonal_acc = float(matrix[t, t])
        final_acc    = float(matrix[-1, t])
        maj_acc      = di_majority[t]
        window_name  = _CT_WINDOWS_DI[t][0]

        if final_acc <= CHANCE_THRESHOLD:
            if t not in di_fresh_cache:
                log(f'    Fresh solo model for DI task {t+1} ({window_name})...')
                di_fresh_cache[t] = run_fresh_solo(
                    di_train_tasks[t], di_test_tasks[t], task_id=t)
            fresh_acc = di_fresh_cache[t]
        else:
            fresh_acc = None

        verdict = assign_verdict(
            diagonal_acc, final_acc,
            di_fresh_cache.get(t, CHANCE_THRESHOLD + 1),
            CHANCE_THRESHOLD,
        )

        rows.append({
            'Setting':      f'DI-{window_name}',
            'Method':       method,
            'Task':         t + 1,
            'Majority_Acc': round(maj_acc, 2),
            'Diagonal_Acc': round(diagonal_acc, 2),
            'Final_Acc':    round(final_acc, 2),
            'Fresh_Acc':    round(fresh_acc, 2) if fresh_acc is not None else 'N/A',
            'Near_Chance':  final_acc <= CHANCE_THRESHOLD,
            'Verdict':      verdict,
        })

# ── report ────────────────────────────────────────────────────────────────────
log('\n' + '='*100)
log('NEAR-CHANCE DIAGNOSTIC TABLE  (CI = class-incremental  |  DI = domain-incremental)')
log('='*100)
hdr = f'{"Setting":<14} {"Method":<18} {"Task":>4} {"Majority":>9} {"Diagonal":>10} {"Final":>7} {"Fresh":>7} {"Verdict":<20}'
log(hdr)
log('-'*100)
for r in rows:
    flag = ' <-- NEAR CHANCE' if r['Near_Chance'] else ''
    log(f'{r["Setting"]:<14} {r["Method"]:<18} {r["Task"]:>4} {r["Majority_Acc"]:>8.1f}% '
        f'{r["Diagonal_Acc"]:>9.1f}% {r["Final_Acc"]:>6.1f}% '
        f'{str(r["Fresh_Acc"]):>6}% {r["Verdict"]:<20}{flag}')
log('='*90)

log('\nKey:')
log('  Diagonal_Acc : accuracy immediately after training that task (shared head)')
log('  Final_Acc    : accuracy after ALL tasks are trained')
log('  Fresh_Acc    : fresh model trained on that task alone (N/A if not near-chance)')
log('  Verdict:')
log('    OK              -- final accuracy above threshold, no concern')
log('    Forgetting      -- diagonal good, final near chance => catastrophic forgetting')
log('    CL Failure      -- diagonal near chance, but solo training succeeds => CL-specific issue')
log('    Task Unlearnable -- even solo training fails => architecture or data issue')

out_csv = LOG_DIR / f'near_chance_analysis_{TIMESTAMP}.csv'
pd.DataFrame(rows).to_csv(out_csv, index=False)

elapsed = time.time() - T0
log(f'\nDone in {elapsed/60:.1f} min')
log(f'Results CSV : {out_csv}')
log(f'Full log    : {LOG_FILE}')
