# ==============================================================================
# RSNA Continual Learning - Patient-level Attention-MIL
# ==============================================================================
# Why this exists:
#   exp_improved_v2.py confirmed that slice-level `any_injury` is too weak a
#   label: it is a PATIENT-level flag stamped on every slice, so most "injured"
#   slices look identical to healthy ones. Training reached ~70% on validation
#   but collapsed to ~50% on held-out test patients (overfitting, not learning).
#
#   This script fixes the label mismatch with Multiple-Instance Learning (MIL).
#   Each PATIENT is a "bag" of slices, and the model predicts ONE label per bag.
#   An attention pooling layer (Ilse et al., 2018) learns to weight the few
#   slices where injury is visible and ignore the rest, which is exactly the
#   problem that capped the per-slice model.
#
# Design:
#   * Bag = one patient; bag label = any_injury.
#   * Encoder: pretrained ResNet-18 (feature vector per slice).
#   * Pooling: gated attention over the bag -> bag embedding -> 2-way logit.
#   * Continual stream: 3-window DOMAIN-INCREMENTAL (Brain/Lung/Soft-tissue),
#     so the CL methods (EWC, Replay, LwF, Combined) still apply, now at the
#     bag level.
#   * Metrics: patient accuracy AND AUC (the natural unit for MIL).
#   * Replay buffer stores whole BAGS (patients), not slices.
#   * Live logging; all output flushed for a foreground terminal.
#
# Usage:
#   conda activate medical_ml
#   $env:DEBUG_RUN = "1"; python src/exp_mil.py     # smoke test
#   python src/exp_mil.py                           # full run
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
LOG_FILE = LOG_DIR / f"experiment_mil_{TIMESTAMP}.log"

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger = logging.getLogger("mil")
_logger.setLevel(logging.INFO)
_logger.addHandler(_fh)


def log(msg):
    print(msg, flush=True)
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

# Hyperparameters
# Scaled up to push patient-level AUC: far more patients (the main lever),
# more slices per bag, a larger encoder, and longer training.
# Best config from the validation-AUC sweep (tune_mil.py, config #3,
# val AUC 68.7): resnet18 encoder, low encoder LR, 40 slices/bag, att dim 128.
EPOCHS_PER_TASK = 1 if DEBUG_RUN else 18
BAGS_PER_STEP = 4 if DEBUG_RUN else 8          # patients per optimisation step
MAX_SLICES_PER_BAG = 6 if DEBUG_RUN else 40    # slices sampled per patient/bag (tuned)
ENCODER = "resnet18"                            # tuned: resnet18 beat resnet34 in the sweep band
ENCODER_LR = 3e-5                               # tuned (more stable than 1e-4/3e-4)
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
EWC_LAMBDA = 200.0
FISHER_BAGS = 8 if DEBUG_RUN else 120
BUFFER_BAGS_PER_TASK = 6 if DEBUG_RUN else 120  # patients kept for replay
LWF_ALPHA, LWF_TEMP = 1.0, 2.0
VAL_FRAC = 0.15

# Use far more data. ~855 injured patients exist; cap healthy near that to keep
# the stream roughly balanced while using ~10x the patients of the first MIL run.
MAX_PATIENTS_PER_CLASS = 8 if DEBUG_RUN else 850
MAX_SLICES_AVAILABLE = 8 if DEBUG_RUN else 60   # slices loaded from disk per patient

CT_WINDOWS = {0: ("Brain", 40, 80), 1: ("Lung", -600, 1500), 2: ("Soft Tissue", 50, 400)}
WINDOW_NAMES = [CT_WINDOWS[i][0] for i in range(3)]

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_AUG = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(8)])


# ============================================================================
# DATA  (one bag = one patient)
# ============================================================================
def apply_ct_window(img, level, width):
    img = img.astype(np.float32)
    lo, hi = level - width / 2, level + width / 2
    w = np.clip(img, lo, hi)
    return (((w - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def load_slice(path, window_type, train=False):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {path}")
    _, lvl, wid = CT_WINDOWS[window_type]
    img = apply_ct_window(img, lvl, wid)
    if img.shape[:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img = np.stack([img, img, img], -1)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    if train:
        t = _AUG(t)
    return _NORM(t)


class Bag:
    """One patient: a list of slice paths + a single label."""
    __slots__ = ("pid", "paths", "label")

    def __init__(self, pid, paths, label):
        self.pid = pid
        self.paths = paths
        self.label = label


def load_bags():
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

    def make_bags(pids):
        bags = []
        for pid in pids:
            pdir = DATA_ROOT / str(pid)
            if not pdir.exists():
                continue
            paths = []
            for sdir in [d for d in pdir.iterdir() if d.is_dir()]:
                for ip in sdir.glob("*.png"):
                    paths.append(str(ip))
                    if len(paths) >= MAX_SLICES_AVAILABLE:
                        break
                if len(paths) >= MAX_SLICES_AVAILABLE:
                    break
            if paths and pid in plabels:
                bags.append(Bag(pid, paths, plabels[pid]))
        return bags

    tr, va, te = make_bags(tr_p), make_bags(va_p), make_bags(te_p)
    log(f"Bags (patients) train/val/test: {len(tr)}/{len(va)}/{len(te)}")
    log(f"Train label balance: {Counter(b.label for b in tr)}")
    return tr, va, te


def sample_bag_tensor(bag, window_type, train=False):
    """Return a [k, 3, H, W] tensor of up-to-MAX_SLICES_PER_BAG slices."""
    paths = bag.paths
    if len(paths) > MAX_SLICES_PER_BAG:
        paths = random.sample(paths, MAX_SLICES_PER_BAG) if train \
            else paths[:MAX_SLICES_PER_BAG]
    return torch.stack([load_slice(p, window_type, train) for p in paths])


# ============================================================================
# ATTENTION-MIL MODEL
# ============================================================================
class AttentionMIL(nn.Module):
    """ResNet-18 encoder + gated attention pooling (Ilse et al., 2018)."""

    def __init__(self, att_dim=128, n_classes=2):
        super().__init__()
        if ENCODER == "resnet34":
            ctor, wts = tv_models.resnet34, getattr(tv_models, "ResNet34_Weights", None)
        else:
            ctor, wts = tv_models.resnet18, getattr(tv_models, "ResNet18_Weights", None)
        try:
            enc = ctor(weights=wts.IMAGENET1K_V1 if wts else None)
        except Exception:
            log("  [warn] pretrained weights unavailable; random init")
            enc = ctor(weights=None)
        feat_dim = enc.fc.in_features
        enc.fc = nn.Identity()
        self.encoder = enc                       # slice -> feat_dim
        self.att_V = nn.Linear(feat_dim, att_dim)
        self.att_U = nn.Linear(feat_dim, att_dim)
        self.att_w = nn.Linear(att_dim, 1)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, bag):
        # bag: [k, 3, H, W]  (one patient)
        h = self.encoder(bag)                    # [k, feat_dim]
        a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))  # [k,1]
        a = torch.softmax(a, dim=0)              # attention weights over slices
        z = (a * h).sum(0, keepdim=True)         # [1, feat_dim] bag embedding
        return self.classifier(z)                # [1, n_classes]


def make_optimizer(model):
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    return torch.optim.AdamW([
        {"params": enc, "lr": ENCODER_LR},
        {"params": head, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


# ============================================================================
# METRICS  (patient-level accuracy + AUC)
# ============================================================================
def auc_score(y_true, y_score):
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ties
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = (sums / counts)[inv]
    r_pos = avg[y_true == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, bags, window_type):
    model.eval()
    y_true, y_score, correct = [], [], 0
    for bag in bags:
        x = sample_bag_tensor(bag, window_type, train=False).to(DEVICE)
        prob = F.softmax(model(x), dim=1)[0, 1].item()
        y_score.append(prob)
        y_true.append(bag.label)
        correct += int((prob >= 0.5) == bag.label)
    acc = 100.0 * correct / max(len(bags), 1)
    return acc, 100.0 * auc_score(y_true, y_score)


# ============================================================================
# EWC / LwF (bag-level)
# ============================================================================
def estimate_fisher(model, bags, n_bags, window_type):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    sample = random.sample(bags, min(n_bags, len(bags)))
    for bag in sample:
        x = sample_bag_tensor(bag, window_type, train=False).to(DEVICE)
        model.zero_grad()
        logp = F.log_softmax(model(x), dim=1)
        F.nll_loss(logp, logp.argmax(1)).backward()
        for n, p in model.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= max(len(sample), 1)
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
# TRAIN ONE TASK  (bag-wise grad accumulation over BAGS_PER_STEP patients)
# ============================================================================
def train_task(model, train_bags, val_bags, window_type, memory=None,
               fisher=None, star=None, ewc_lambda=0.0, teacher=None, tag=""):
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)

    # class weights for balanced bag loss
    counts = Counter(b.label for b in train_bags)
    cw = torch.tensor([1.0 / max(counts.get(0, 1), 1), 1.0 / max(counts.get(1, 1), 1)],
                      device=DEVICE)
    cw = cw / cw.sum() * 2

    best_val, best_state, bad = -1.0, None, 0
    for epoch in range(EPOCHS_PER_TASK):
        model.train()
        order = list(range(len(train_bags)))
        random.shuffle(order)
        run_loss, nb = 0.0, 0
        opt.zero_grad()
        for step, bi in enumerate(order):
            bag = train_bags[bi]
            x = sample_bag_tensor(bag, window_type, train=True).to(DEVICE)
            y = torch.tensor([bag.label], device=DEVICE)
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                out = model(x)
                loss = F.cross_entropy(out, y, weight=cw)
                if memory:
                    mbag = random.choice(memory)
                    mx = sample_bag_tensor(mbag, mbag_window(mbag), train=True).to(DEVICE)
                    my = torch.tensor([mbag.label], device=DEVICE)
                    loss = 0.5 * loss + 0.5 * F.cross_entropy(model(mx), my, weight=cw)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(x)
                    loss = loss + LWF_ALPHA * lwf_loss(out, tl, LWF_TEMP)
                if fisher is not None and star is not None and ewc_lambda > 0:
                    loss = loss + ewc_lambda * ewc_penalty(model, fisher, star)
            (loss / BAGS_PER_STEP).backward()
            run_loss += float(loss)
            nb += 1
            if (step + 1) % BAGS_PER_STEP == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()

        val_acc, val_auc = evaluate(model, val_bags, window_type)
        sched.step(val_auc)
        log(f"    [{tag}] epoch {epoch+1}/{EPOCHS_PER_TASK} | loss {run_loss/max(nb,1):.4f} "
            f"| val acc {val_acc:.1f}% AUC {val_auc:.1f}")
        if val_auc > best_val:
            best_val, best_state, bad = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                log(f"    [{tag}] early stop at epoch {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# memory bags remember which window they were stored under
_BAG_WINDOW = {}


def mbag_window(bag):
    return _BAG_WINDOW.get(bag.pid, 0)


# ============================================================================
# DRIVER
# ============================================================================
def run_method(method, train_bags, val_bags, test_bags, ewc_lambda=EWC_LAMBDA):
    model = AttentionMIL().to(DEVICE)
    T_ = 3
    R = np.zeros((T_, T_))        # accuracy matrix
    A = np.zeros((T_, T_))        # AUC matrix
    memory, fisher, star, teacher = [], None, None, None

    for t in range(T_):
        log(f"\n  --- Task {t+1}/{T_}: {WINDOW_NAMES[t]} window ({method}) ---")
        use_mem = memory if method in ("replay", "combined") else None
        use_fisher = fisher if method in ("ewc", "combined") else None
        use_star = star if method in ("ewc", "combined") else None
        use_lambda = ewc_lambda if method in ("ewc", "combined") else 0.0
        use_teacher = teacher if method == "lwf" else None

        train_task(model, train_bags, val_bags, t, memory=use_mem,
                   fisher=use_fisher, star=use_star, ewc_lambda=use_lambda,
                   teacher=use_teacher, tag=method)

        if method in ("ewc", "combined"):
            log("    estimating Fisher information...")
            fisher = estimate_fisher(model, train_bags, FISHER_BAGS, t)
            star = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if method == "lwf":
            teacher = copy.deepcopy(model).eval()
            for p in teacher.parameters():
                p.requires_grad = False
        if method in ("replay", "combined"):
            picks = random.sample(train_bags, min(BUFFER_BAGS_PER_TASK, len(train_bags)))
            for b in picks:
                _BAG_WINDOW[b.pid] = t
            memory += picks
            log(f"    replay buffer: {len(memory)} bags")

        for j in range(T_):
            R[t, j], A[t, j] = evaluate(model, test_bags, j)
        log(f"    acc R[{t+1}] = {np.round(R[t],1).tolist()} | AUC = {np.round(A[t],1).tolist()}")
    return R, A


def avg(M):
    return float(M[-1].mean())


def forgetting(M):
    T_ = M.shape[0]
    return float(np.mean([M[:T_-1, j].max() - M[T_-1, j] for j in range(T_-1)])) if T_ > 1 else 0.0


def main():
    set_seed(42)
    log("=" * 72)
    log("PATIENT-LEVEL ATTENTION-MIL - 3-window domain-incremental")
    log(f"Device: {DEVICE} | DEBUG_RUN={DEBUG_RUN}")
    log(f"Encoder {ENCODER} | epochs/task {EPOCHS_PER_TASK} | bags/step {BAGS_PER_STEP} | "
        f"slices/bag {MAX_SLICES_PER_BAG} | patients/class<={MAX_PATIENTS_PER_CLASS} | "
        f"enc LR {ENCODER_LR} head LR {HEAD_LR}")
    log("=" * 72)

    tr, va, te = load_bags()
    methods = ["baseline", "ewc", "replay", "lwf", "combined"]
    results = {}
    for m in methods:
        log(f"\n{'='*72}\nMETHOD: {m}\n{'='*72}")
        _BAG_WINDOW.clear()
        R, A = run_method(m, tr, va, te)
        results[m] = {"R": R, "A": A, "acc": avg(R), "acc_fgt": forgetting(R),
                      "auc": avg(A), "auc_fgt": forgetting(A)}
        log(f"  >> {m}: patient ACC {results[m]['acc']:.1f}% (FGT {results[m]['acc_fgt']:.1f}) "
            f"| AUC {results[m]['auc']:.1f} (FGT {results[m]['auc_fgt']:.1f})")

    out_csv = LOG_DIR / f"results_mil_{TIMESTAMP}.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Method,Patient_Acc,Acc_Fgt,Patient_AUC,AUC_Fgt\n")
        for m, r in results.items():
            f.write(f"{m},{r['acc']:.4f},{r['acc_fgt']:.4f},{r['auc']:.4f},{r['auc_fgt']:.4f}\n")
    np.savez(LOG_DIR / f"matrices_mil_{TIMESTAMP}.npz",
             **{f"{m}_acc": r["R"] for m, r in results.items()},
             **{f"{m}_auc": r["A"] for m, r in results.items()})

    log(f"\n{'='*72}\nSUMMARY - Attention-MIL (patient-level)\n{'='*72}")
    log(f"{'Method':<12}{'Patient ACC':>13}{'ACC FGT':>10}{'AUC':>9}{'AUC FGT':>10}")
    for m, r in results.items():
        log(f"{m:<12}{r['acc']:>12.1f}%{r['acc_fgt']:>9.1f}%{r['auc']:>8.1f}{r['auc_fgt']:>9.1f}")
    log(f"\nResults: {out_csv}")
    log("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"Total time: {time.time()-t0:.1f}s")
