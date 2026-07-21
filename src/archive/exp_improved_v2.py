# ==============================================================================
# RSNA Continual Learning - Improved Experiment v2
# ==============================================================================
# Goal: a learnable continual-learning setup that reaches clinically useful
# accuracy (target 78-98%), unlike the earlier 2-task/1-class-each design that
# was capped at chance by construction.
#
# Design choices (each is a deliberate accuracy lever):
#   * 3-window DOMAIN-INCREMENTAL stream. BOTH classes appear in EVERY task, so
#     each task is a genuine binary problem the model can score above chance on.
#       Task 1: Brain window   (WL=40,   WW=80)
#       Task 2: Lung window    (WL=-600, WW=1500)
#       Task 3: Soft tissue    (WL=50,   WW=400)
#   * Pretrained ResNet-18, FULLY fine-tuned (all layers) at a low LR.
#   * Train-time augmentation (flip / rotation / affine jitter).
#   * Class balancing via WeightedRandomSampler.
#   * Validation split + ReduceLROnPlateau + early stopping (avoids the
#     overfit-to-one-task collapse seen before).
#   * Patient-level aggregation (mean slice probability) reported as the primary,
#     clinically meaningful metric alongside slice-level accuracy.
#   * Herding exemplar selection rewritten to extract features in BATCHES with a
#     hard candidate cap, a time budget, and a guaranteed random fallback. This
#     is what stalled the previous run; it can no longer hang.
#   * Live logging: every line is printed and flushed so a foreground terminal
#     shows progress in real time.
#
# Usage:
#   conda activate medical_ml
#   $env:DEBUG_RUN = "1"; python src/exp_improved_v2.py      # fast smoke test
#   python src/exp_improved_v2.py                            # full run
# ==============================================================================

import os
import copy
import time
import random
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as tv_models
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")

# ============================================================================
# CONFIG
# ============================================================================
DEBUG_RUN = os.environ.get("DEBUG_RUN", "0") == "1"
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"experiment_improved_v2_{TIMESTAMP}.log"

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger = logging.getLogger("improved_v2")
_logger.setLevel(logging.INFO)
_logger.addHandler(_fh)


def log(msg):
    print(msg, flush=True)          # live terminal output
    _logger.info(msg)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not FORCE_CPU:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


DEVICE = torch.device("cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

DATA_DIR = Path("data")
DATA_ROOT = DATA_DIR / "RSNA2023ProcessedImages"
LABELS_CSV = DATA_DIR / "train.csv"
TARGET_SIZE = (224, 224)

# ---- accuracy-oriented hyperparameters ----
EPOCHS_PER_TASK = 1 if DEBUG_RUN else 12       # epochs (not raw iters) per task
BATCH_SIZE = 8 if DEBUG_RUN else 32
HEAD_LR = 1e-3                                  # classifier head LR
BACKBONE_LR = 1e-4                              # fine-tune the backbone slowly
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 3                         # epochs w/o val improvement
EWC_LAMBDA = 200.0
FISHER_SAMPLES = 16 if DEBUG_RUN else 256
BUFFER_PER_TASK = 40 if DEBUG_RUN else 600
LWF_ALPHA, LWF_TEMP = 1.0, 2.0
VAL_FRAC = 0.15

# ---- herding safety limits (these fix the previous hang) ----
HERDING_MAX_CANDIDATES = 1500                   # cap images we featurise
HERDING_TIME_BUDGET_S = 120                     # abort -> random fallback

# ---- data sampling (more data than before) ----
MAX_PATIENTS_PER_CLASS = 6 if DEBUG_RUN else 160
MAX_IMAGES_PER_PATIENT = 4 if DEBUG_RUN else 24

CT_WINDOWS = {0: ("Brain", 40, 80), 1: ("Lung", -600, 1500), 2: ("Soft Tissue", 50, 400)}
WINDOW_NAMES = [CT_WINDOWS[i][0] for i in range(3)]


# ============================================================================
# DATA
# ============================================================================
def apply_ct_window(img, level, width):
    img = img.astype(np.float32)
    lo, hi = level - width / 2, level + width / 2
    w = np.clip(img, lo, hi)
    return (((w - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def load_image(img_path, window_type):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load: {img_path}")
    _, lvl, wid = CT_WINDOWS[window_type]
    img = apply_ct_window(img, lvl, wid)
    if img.shape[:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img = np.stack([img, img, img], axis=-1)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ImageNet normalisation (the backbone was pretrained with these stats)
_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_AUG = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])


class WindowDataset(Dataset):
    """Slices under one CT window. `train` toggles augmentation."""

    def __init__(self, paths, labels, patients, window_type, train=False):
        self.paths = paths
        self.labels = labels
        self.patients = patients
        self.window_type = window_type
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image(self.paths[idx], self.window_type)
        if self.train:
            img = _AUG(img)
        return _NORM(img), self.labels[idx]


class MemoryDataset(Dataset):
    """Replay buffer: (path, label, window_type) tuples, ImageNet-normalised."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, wt = self.items[idx]
        return _NORM(load_image(path, wt)), label


def load_data():
    df = pd.read_csv(LABELS_CSV)
    plabels = {int(r["patient_id"]): int(r["any_injury"])
               for _, r in df[["patient_id", "any_injury"]].dropna().iterrows()}
    by_class = defaultdict(list)
    for pid, lab in plabels.items():
        by_class[lab].append(pid)

    rng = random.Random(42)
    tr_p, va_p, te_p = [], [], []
    for lab, pats in by_class.items():
        rng.shuffle(pats)
        pats = pats[:MAX_PATIENTS_PER_CLASS]
        n_te = max(1, int(len(pats) * 0.2))
        n_va = max(1, int(len(pats) * VAL_FRAC))
        te_p += pats[:n_te]
        va_p += pats[n_te:n_te + n_va]
        tr_p += pats[n_te + n_va:]

    def collect(pids):
        paths, labels, patients = [], [], []
        for pid in pids:
            pdir = DATA_ROOT / str(pid)
            if not pdir.exists():
                continue
            imgs = []
            for sdir in [d for d in pdir.iterdir() if d.is_dir()]:
                for ip in sdir.glob("*.png"):
                    imgs.append(str(ip))
                    if len(imgs) >= MAX_IMAGES_PER_PATIENT:
                        break
                if len(imgs) >= MAX_IMAGES_PER_PATIENT:
                    break
            lab = plabels.get(pid)
            if lab is None:
                continue
            paths += imgs
            labels += [lab] * len(imgs)
            patients += [pid] * len(imgs)
        return paths, labels, patients

    tr = collect(tr_p)
    va = collect(va_p)
    te = collect(te_p)
    log(f"Patients  train/val/test: {len(tr_p)}/{len(va_p)}/{len(te_p)}")
    log(f"Slices    train/val/test: {len(tr[0])}/{len(va[0])}/{len(te[0])}")
    log(f"Train class balance: {Counter(tr[1])}")
    return tr, va, te


# ============================================================================
# MODEL (full fine-tune)
# ============================================================================
def build_model():
    try:
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        log("  [warn] pretrained weights unavailable; random init")
        model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model  # all layers trainable


def make_optimizer(model):
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([
        {"params": backbone, "lr": BACKBONE_LR},
        {"params": head, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


# ============================================================================
# METRICS  (slice-level and patient-level)
# ============================================================================
@torch.no_grad()
def evaluate(model, ds):
    """Return (slice_acc, patient_acc). Patient-level = mean injury probability
    over the patient's slices, thresholded at 0.5."""
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    correct = total = 0
    pat_prob = defaultdict(list)
    pat_true = {}
    i = 0
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        prob = F.softmax(model(imgs), dim=1)[:, 1].cpu()
        preds = (prob >= 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.numel()
        for k in range(labels.numel()):
            pid = ds.patients[i + k]
            pat_prob[pid].append(float(prob[k]))
            pat_true[pid] = int(labels[k])
        i += labels.numel()
    slice_acc = 100.0 * correct / max(total, 1)
    pc = sum(int((np.mean(p) >= 0.5) == pat_true[pid]) for pid, p in pat_prob.items())
    pat_acc = 100.0 * pc / max(len(pat_prob), 1)
    return slice_acc, pat_acc


# ============================================================================
# EWC / LwF
# ============================================================================
def estimate_fisher(model, ds, n_samples):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    seen = 0
    for imgs, _ in loader:
        if seen >= n_samples:
            break
        imgs = imgs.to(DEVICE)
        model.zero_grad()
        logp = F.log_softmax(model(imgs), dim=1)
        F.nll_loss(logp, logp.argmax(1)).backward()
        for n, p in model.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += p.grad.detach() ** 2
        seen += 1
    for n in fisher:
        fisher[n] /= max(seen, 1)
    return fisher


def ewc_penalty(model, fisher, star):
    loss = torch.tensor(0.0, device=DEVICE)
    for n, p in model.named_parameters():
        if n in fisher:
            loss = loss + (fisher[n] * (p - star[n]) ** 2).sum()
    return 0.5 * loss


def lwf_loss(student, teacher_logits, T_=2.0):
    s = F.log_softmax(student / T_, dim=1)
    t = F.softmax(teacher_logits / T_, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T_ * T_)


# ============================================================================
# HERDING (batched, capped, time-bounded -> cannot hang)
# ============================================================================
@torch.no_grad()
def select_exemplars(model, paths, labels, window_type, buffer_size):
    n = len(paths)
    idx_all = list(range(n))
    random.shuffle(idx_all)

    def random_pick():
        chosen = idx_all[:buffer_size]
        return [(paths[i], labels[i], window_type) for i in chosen]

    if n <= buffer_size:
        return [(paths[i], labels[i], window_type) for i in range(n)]

    # Featurise only a capped pool of candidates, in batches, under a time budget.
    cand = idx_all[:HERDING_MAX_CANDIDATES]
    feat_model = nn.Sequential(*list(model.children())[:-1]).to(DEVICE).eval()
    t0 = time.time()
    feats, kept = [], []
    B = 64
    for s in range(0, len(cand), B):
        if time.time() - t0 > HERDING_TIME_BUDGET_S:
            log("    [herding] time budget exceeded -> random fallback")
            return random_pick()
        batch_idx = cand[s:s + B]
        imgs = torch.stack([_NORM(load_image(paths[i], window_type)) for i in batch_idx]).to(DEVICE)
        f = feat_model(imgs).flatten(1).cpu().numpy()
        feats.append(f)
        kept.extend(batch_idx)
    feats = np.concatenate(feats, 0)

    by_class = defaultdict(list)
    for row, i in enumerate(kept):
        by_class[labels[i]].append(row)

    chosen = []
    per_class = max(1, buffer_size // max(len(by_class), 1))
    for cls, rows in by_class.items():
        cf = feats[rows]
        mean = cf.mean(0)
        running = np.zeros_like(mean)
        picked = []
        for k in range(min(per_class, len(rows))):
            scores = np.linalg.norm(mean - (running + cf) / (k + 1), axis=1)
            order = np.argsort(scores)
            for o in order:
                if rows[o] not in picked:
                    picked.append(rows[o])
                    running += cf[o]
                    break
        chosen.extend(kept[r] for r in picked)
    return [(paths[i], labels[i], window_type) for i in chosen]


# ============================================================================
# TRAIN ONE TASK (epoch-based, val early-stop, scheduler, balanced sampler)
# ============================================================================
def train_task(model, train_ds, val_ds, memory=None, fisher=None, star=None,
               ewc_lambda=0.0, teacher=None, tag=""):
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    # class-balanced sampler
    counts = Counter(train_ds.labels)
    w = [1.0 / counts[l] for l in train_ds.labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    mem_iter = None
    if memory:
        mem_loader = DataLoader(MemoryDataset(memory), batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0)

        def cyc(dl):
            while True:
                for b in dl:
                    yield b
        mem_iter = cyc(mem_loader)

    best_val, best_state, bad = -1.0, None, 0
    for epoch in range(EPOCHS_PER_TASK):
        model.train()
        run_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                out = model(imgs)
                loss = F.cross_entropy(out, labels)
                if mem_iter is not None:
                    m_imgs, m_labels = next(mem_iter)
                    m_imgs, m_labels = m_imgs.to(DEVICE), m_labels.to(DEVICE)
                    loss = 0.5 * loss + 0.5 * F.cross_entropy(model(m_imgs), m_labels)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(imgs)
                    loss = loss + LWF_ALPHA * lwf_loss(out, t_logits, LWF_TEMP)
                if fisher is not None and star is not None and ewc_lambda > 0:
                    loss = loss + ewc_lambda * ewc_penalty(model, fisher, star)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss)

        val_slice, val_pat = evaluate(model, val_ds)
        sched.step(val_pat)
        log(f"    [{tag}] epoch {epoch+1}/{EPOCHS_PER_TASK} | loss {run_loss/max(len(loader),1):.4f} "
            f"| val slice {val_slice:.1f}% patient {val_pat:.1f}%")
        if val_pat > best_val:
            best_val, best_state, bad = val_pat, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                log(f"    [{tag}] early stop at epoch {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================================
# DRIVER
# ============================================================================
def run_method(method, tasks_train, tasks_val, tasks_test, ewc_lambda=EWC_LAMBDA):
    model = build_model().to(DEVICE)
    T_ = len(tasks_train)
    R = np.zeros((T_, T_))
    Rpat = np.zeros((T_, T_))
    memory, fisher, star, teacher = [], None, None, None

    for t in range(T_):
        log(f"\n  --- Task {t+1}/{T_}: {WINDOW_NAMES[t]} window ({method}) ---")
        use_mem = memory if method in ("replay", "combined") else None
        use_fisher = fisher if method in ("ewc", "combined") else None
        use_star = star if method in ("ewc", "combined") else None
        use_lambda = ewc_lambda if method in ("ewc", "combined") else 0.0
        use_teacher = teacher if method == "lwf" else None

        train_task(model, tasks_train[t], tasks_val[t], memory=use_mem,
                   fisher=use_fisher, star=use_star, ewc_lambda=use_lambda,
                   teacher=use_teacher, tag=method)

        if method in ("ewc", "combined"):
            log("    estimating Fisher information...")
            fisher = estimate_fisher(model, tasks_train[t], FISHER_SAMPLES)
            star = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if method == "lwf":
            teacher = copy.deepcopy(model).eval()
            for p in teacher.parameters():
                p.requires_grad = False
        if method in ("replay", "combined"):
            log("    updating replay buffer (herding)...")
            ds = tasks_train[t]
            memory += select_exemplars(model, ds.paths, ds.labels, ds.window_type, BUFFER_PER_TASK)
            log(f"    buffer size: {len(memory)}")

        for j in range(T_):
            R[t, j], Rpat[t, j] = evaluate(model, tasks_test[j])
        log(f"    slice R[{t+1}] = {np.round(R[t],1).tolist()}")
        log(f"    patient R[{t+1}] = {np.round(Rpat[t],1).tolist()}")
    return R, Rpat


def avg_acc(R):
    return float(R[-1].mean())


def forgetting(R):
    T_ = R.shape[0]
    if T_ < 2:
        return 0.0
    return float(np.mean([R[:T_-1, j].max() - R[T_-1, j] for j in range(T_-1)]))


def make_window_tasks(split, train=False):
    paths, labels, patients = split
    return [WindowDataset(paths, labels, patients, wt, train=train) for wt in range(3)]


def main():
    set_seed(42)
    log("=" * 72)
    log("IMPROVED EXPERIMENT v2 - 3-window domain-incremental, full fine-tune")
    log(f"Device: {DEVICE} | DEBUG_RUN={DEBUG_RUN}")
    log(f"Epochs/task: {EPOCHS_PER_TASK} | batch: {BATCH_SIZE} | "
        f"backbone LR {BACKBONE_LR} head LR {HEAD_LR} | buffer/task {BUFFER_PER_TASK}")
    log("=" * 72)

    tr, va, te = load_data()
    tasks_train = make_window_tasks(tr, train=True)
    tasks_val = make_window_tasks(va, train=False)
    tasks_test = make_window_tasks(te, train=False)

    methods = ["baseline", "ewc", "replay", "lwf", "combined"]
    results = {}
    for m in methods:
        log(f"\n{'='*72}\nMETHOD: {m}\n{'='*72}")
        R, Rpat = run_method(m, tasks_train, tasks_val, tasks_test)
        results[m] = {
            "R": R, "Rpat": Rpat,
            "slice_acc": avg_acc(R), "slice_fgt": forgetting(R),
            "pat_acc": avg_acc(Rpat), "pat_fgt": forgetting(Rpat),
        }
        log(f"  >> {m}: slice ACC {results[m]['slice_acc']:.1f}% (FGT {results[m]['slice_fgt']:.1f}) "
            f"| patient ACC {results[m]['pat_acc']:.1f}% (FGT {results[m]['pat_fgt']:.1f})")

    out_csv = LOG_DIR / f"results_improved_v2_{TIMESTAMP}.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Method,Slice_Acc,Slice_Fgt,Patient_Acc,Patient_Fgt\n")
        for m, r in results.items():
            f.write(f"{m},{r['slice_acc']:.4f},{r['slice_fgt']:.4f},"
                    f"{r['pat_acc']:.4f},{r['pat_fgt']:.4f}\n")
    np.savez(LOG_DIR / f"matrices_improved_v2_{TIMESTAMP}.npz",
             **{f"{m}_slice": r["R"] for m, r in results.items()},
             **{f"{m}_patient": r["Rpat"] for m, r in results.items()})

    log(f"\n{'='*72}\nSUMMARY v2 (3-window domain-incremental)\n{'='*72}")
    log(f"{'Method':<12}{'Slice ACC':>11}{'Slice FGT':>11}{'Patient ACC':>13}{'Patient FGT':>13}")
    for m, r in results.items():
        log(f"{m:<12}{r['slice_acc']:>10.1f}%{r['slice_fgt']:>10.1f}%"
            f"{r['pat_acc']:>12.1f}%{r['pat_fgt']:>12.1f}%")
    log(f"\nResults: {out_csv}")
    log("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"Total time: {time.time()-t0:.1f}s")
