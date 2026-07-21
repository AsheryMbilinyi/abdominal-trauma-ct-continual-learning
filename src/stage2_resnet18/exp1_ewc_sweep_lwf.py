# ==============================================================================
# RSNA Continual Learning - Improved Experiment (next iteration)
# ==============================================================================
# Implements the six modifications motivated by the report's analysis:
#   1. Pretrained ResNet-18 backbone (replaces the over-parameterised custom CNN)
#   2. EWC lambda sweep over {10, 50, 100, 500, 1000}
#   3. Balanced replay loss (equal old/new contribution per step)
#   4. Larger replay buffer + herding exemplar selection
#   5. Knowledge distillation (Learning without Forgetting, LwF)
#   6. Patient/series-level label aggregation for evaluation
#
# Data loading, windowing and dataset classes are reused from the
# class-incremental experiment so behaviour stays consistent.
#
# Usage:
#   conda activate medical_ml
#   $env:DEBUG_RUN = "1"; python src/exp_improved.py      # fast smoke test
#   python src/exp_improved.py                            # full run
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
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models

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
LOG_FILE = LOG_DIR / f"experiment_improved_{TIMESTAMP}.log"

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger = logging.getLogger("improved")
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
TARGET_SIZE = (224, 224)  # ResNet-18 native input

# Hyperparameters
ITERS_PER_TASK = 3 if DEBUG_RUN else 1000
BATCH_SIZE = 8 if DEBUG_RUN else 32
LEARNING_RATE = 1e-3
EWC_LAMBDA_GRID = [10.0] if DEBUG_RUN else [10.0, 50.0, 100.0, 500.0, 1000.0]  # Mod 2
FISHER_SAMPLES = 10 if DEBUG_RUN else 300
BUFFER_PER_TASK = 20 if DEBUG_RUN else 800  # Mod 4: larger buffer
HERDING = True                              # Mod 4: herding selection
LWF_ALPHA = 1.0                             # Mod 5
LWF_TEMP = 2.0                              # Mod 5
MAX_PATIENTS = 4 if DEBUG_RUN else 100
MAX_IMAGES = 5 if DEBUG_RUN else 20


# ============================================================================
# DATA LAYER (self-contained; mirrors exp_class_incremental.py)
# ============================================================================
CT_WINDOWS = {
    0: ("Brain", 40, 80),
    1: ("Lung", -600, 1500),
    2: ("Soft Tissue", 50, 400),
}


def apply_ct_window(img, level, width):
    img = img.astype(np.float32)
    lo, hi = level - width // 2, level + width // 2
    w = np.clip(img, lo, hi)
    return (((w - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def load_image_tensor(img_path, target_size, transform=None, window_type=None):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load: {img_path}")
    if window_type is not None and window_type in CT_WINDOWS:
        _, lvl, wid = CT_WINDOWS[window_type]
        img = apply_ct_window(img, lvl, wid)
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = np.stack([img, img, img], axis=-1)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return transform(t) if transform else t


class RSNADataset(Dataset):
    def __init__(self, image_paths, labels, target_size=(224, 224), window_type=None):
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size
        self.window_type = window_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return (load_image_tensor(self.image_paths[idx], self.target_size,
                                  None, self.window_type),
                self.labels[idx])


class SubDataset(Dataset):
    """Subset containing only the specified labels (class-incremental tasks)."""

    def __init__(self, original, sub_labels):
        self.original = original
        self.indices = [i for i, l in enumerate(original.labels) if l in sub_labels]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original[self.indices[idx]]

    def get_path_and_label(self, idx):
        oi = self.indices[idx]
        return self.original.image_paths[oi], self.original.labels[oi]


class MemorySetDataset(Dataset):
    """Replay buffer dataset of (path, label, window_type) tuples."""

    def __init__(self, memory_sets, target_size, transform=None):
        self.memory_sets = memory_sets
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.memory_sets)

    def __getitem__(self, idx):
        path, label, wt = self.memory_sets[idx]
        return load_image_tensor(path, self.target_size, self.transform, wt), label


def load_rsna_data(data_root, labels_csv, test_split=0.2, max_patients_per_class=None,
                   max_images_per_patient=None, seed=42):
    data_root, labels_csv = Path(data_root), Path(labels_csv)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    df = pd.read_csv(labels_csv)
    patient_labels = {int(r["patient_id"]): int(r["any_injury"])
                      for _, r in df[["patient_id", "any_injury"]].dropna().iterrows()}
    label_to_patients = defaultdict(list)
    for pid, lab in patient_labels.items():
        label_to_patients[lab].append(pid)

    rng = random.Random(seed)
    train_p, test_p = [], []
    for lab, pats in label_to_patients.items():
        rng.shuffle(pats)
        if max_patients_per_class is not None:
            pats = pats[:max_patients_per_class]
        cut = int(len(pats) * (1.0 - test_split))
        train_p += pats[:cut]
        test_p += pats[cut:]

    def collect(pids):
        paths, labels = [], []
        for pid in pids:
            pdir = data_root / str(pid)
            if not pdir.exists():
                continue
            imgs = []
            for sdir in [d for d in pdir.iterdir() if d.is_dir()]:
                for ip in sdir.glob("*.png"):
                    imgs.append(str(ip))
                    if max_images_per_patient and len(imgs) >= max_images_per_patient:
                        break
                if max_images_per_patient and len(imgs) >= max_images_per_patient:
                    break
            lab = patient_labels.get(pid)
            if lab is None:
                continue
            paths += imgs
            labels += [lab] * len(imgs)
        return paths, labels

    trp, trl = collect(train_p)
    tep, tel = collect(test_p)
    if not trp or not tep:
        raise ValueError("Train/test split produced no images.")
    log(f"Loaded {len(trp)} train / {len(tep)} test images")
    log(f"Class distribution (train): {Counter(trl)}")
    return trp, trl, tep, tel


def create_task_datasets(base_dataset, num_tasks):
    """Split a dataset into class-incremental tasks."""
    all_labels = sorted(set(base_dataset.labels))
    per = max(1, len(all_labels) // num_tasks)
    tasks = []
    for t in range(num_tasks):
        sub = all_labels[t * per:(t + 1) * per]
        ds = SubDataset(base_dataset, sub)
        tasks.append(ds)
        log(f"  Task {t+1}: classes {sub}, {len(ds)} samples")
    return tasks


# ============================================================================
# MODIFICATION 1: Pretrained ResNet-18 backbone
# ============================================================================
def build_resnet18(num_classes=2, freeze_early=True):
    """ResNet-18 with ImageNet weights; early layers optionally frozen."""
    try:
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        model = tv_models.resnet18(weights=weights)
    except Exception:
        # Older torchvision / offline: fall back to random init
        log("  [warn] pretrained weights unavailable; using random init")
        model = tv_models.resnet18(weights=None)
    if freeze_early:
        # Freeze stem + first two residual stages; fine-tune layer3/4 + fc.
        for name, p in model.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def named_trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


# ============================================================================
# EVALUATION (+ MOD 6: patient/series-level aggregation)
# ============================================================================
@torch.no_grad()
def evaluate(model, dataset, paths=None, patient_of=None, aggregate=False):
    """Return slice-level accuracy, and optionally patient-level accuracy.

    Mod 6: when `aggregate` and `patient_of` are given, predictions are pooled
    per patient by majority vote, then scored against the patient label.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    correct = total = 0
    per_patient_pred = defaultdict(list)
    per_patient_true = {}
    idx = 0
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.numel()
        if aggregate and paths is not None and patient_of is not None:
            for k in range(labels.numel()):
                pid = patient_of(paths[idx + k])
                per_patient_pred[pid].append(int(preds[k]))
                per_patient_true[pid] = int(labels[k])
            idx += labels.numel()
    slice_acc = 100.0 * correct / max(total, 1)
    if aggregate and per_patient_pred:
        pc = pt = 0
        for pid, preds in per_patient_pred.items():
            maj = Counter(preds).most_common(1)[0][0]
            pc += int(maj == per_patient_true[pid])
            pt += 1
        return slice_acc, 100.0 * pc / max(pt, 1)
    return slice_acc, None


# ============================================================================
# MOD 5: Knowledge distillation (LwF)
# ============================================================================
def lwf_loss(student_logits, teacher_logits, T=2.0):
    """KL(soft_teacher || soft_student) scaled by T^2 (Hinton distillation)."""
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


# ============================================================================
# FISHER INFORMATION + EWC PENALTY (Eq. 3-4 in the report)
# ============================================================================
def estimate_fisher(model, dataset, n_samples):
    fisher = {n: torch.zeros_like(p) for n, p in named_trainable(model)}
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    seen = 0
    for imgs, labels in loader:
        if seen >= n_samples:
            break
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()
        logp = F.log_softmax(model(imgs), dim=1)
        # sample label from the model's own predictive distribution (true Fisher)
        loss = F.nll_loss(logp, logp.argmax(1))
        loss.backward()
        for n, p in named_trainable(model):
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        seen += 1
    for n in fisher:
        fisher[n] /= max(seen, 1)
    return fisher


def ewc_penalty(model, fisher, star_params):
    loss = torch.tensor(0.0, device=DEVICE)
    for n, p in named_trainable(model):
        if n in fisher:
            loss = loss + (fisher[n] * (p - star_params[n]) ** 2).sum()
    return 0.5 * loss


# ============================================================================
# MOD 4: herding exemplar selection
# ============================================================================
@torch.no_grad()
def select_exemplars(model, task_paths, task_labels, window_type, buffer_size):
    """Pick buffer_size exemplars. Herding => closest to class feature mean."""
    if not HERDING or len(task_paths) <= buffer_size:
        idx = list(range(len(task_paths)))
        random.shuffle(idx)
        idx = idx[:buffer_size]
        return [(task_paths[i], task_labels[i], window_type) for i in idx]

    # feature extractor = backbone without final fc
    feat_model = nn.Sequential(*list(model.children())[:-1]).to(DEVICE).eval()
    feats, kept = [], []
    for i, pth in enumerate(task_paths):
        t = load_image_tensor(pth, TARGET_SIZE, None, window_type).unsqueeze(0).to(DEVICE)
        f = feat_model(t).flatten(1).cpu().numpy()[0]
        feats.append(f)
        kept.append(i)
    feats = np.array(feats)
    by_class = defaultdict(list)
    for j, i in enumerate(kept):
        by_class[task_labels[i]].append(j)

    chosen = []
    per_class = max(1, buffer_size // max(len(by_class), 1))
    for cls, members in by_class.items():
        cf = feats[members]
        mean = cf.mean(0)
        # greedy herding: iteratively pick sample reducing distance to mean
        selected, running = [], np.zeros_like(mean)
        for k in range(min(per_class, len(members))):
            scores = np.linalg.norm(mean - (running + cf) / (k + 1), axis=1)
            for s in np.argsort(scores):
                if members[s] not in selected:
                    selected.append(members[s])
                    running += cf[s]
                    break
        chosen.extend(selected)
    return [(task_paths[i], task_labels[i], window_type) for i in chosen]


# ============================================================================
# TRAINING (Baseline / EWC / Replay / Combined / LwF) with MOD 3 balanced replay
# ============================================================================
def train_task(model, task_dataset, iters, lr, batch_size,
               memory=None, fisher=None, star=None, ewc_lambda=0.0,
               teacher=None):
    """One task's training. Replay (Mod 3) draws a balanced memory batch each
    step; EWC adds the Fisher penalty; LwF adds a distillation term."""
    model.train()
    opt = torch.optim.Adam(trainable_params(model), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=False)
    mem_loader = None
    if memory:
        mem_ds = MemorySetDataset(memory, TARGET_SIZE, None)
        mem_loader = DataLoader(mem_ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, drop_last=False)

    def cycle(dl):
        while True:
            for b in dl:
                yield b

    new_iter = cycle(loader)
    mem_iter = cycle(mem_loader) if mem_loader else None

    for it in range(iters):
        imgs, labels = next(new_iter)
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            out = model(imgs)
            loss_new = F.cross_entropy(out, labels)

            # Mod 3: balanced replay — equal weight to memory batch
            loss_mem = torch.tensor(0.0, device=DEVICE)
            if mem_iter is not None:
                m_imgs, m_labels = next(mem_iter)
                m_imgs, m_labels = m_imgs.to(DEVICE), m_labels.to(DEVICE)
                loss_mem = F.cross_entropy(model(m_imgs), m_labels)
                loss = 0.5 * loss_new + 0.5 * loss_mem
            else:
                loss = loss_new

            # Mod 5: LwF distillation against the frozen previous model
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                loss = loss + LWF_ALPHA * lwf_loss(out, t_logits, LWF_TEMP)

            # EWC penalty
            if fisher is not None and star is not None and ewc_lambda > 0:
                loss = loss + ewc_lambda * ewc_penalty(model, fisher, star)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (it + 1) % max(1, iters // 4) == 0:
            log(f"    iter {it+1}/{iters} | loss {loss.item():.4f}")
    return model


# ============================================================================
# CONTINUAL-LEARNING DRIVER
# ============================================================================
def run_method(method, task_train, task_test, num_tasks, ewc_lambda,
               test_paths=None, patient_of=None):
    """Train sequentially with the chosen method; return accuracy matrix R."""
    model = build_resnet18().to(DEVICE)
    R = np.zeros((num_tasks, num_tasks))
    Rpat = np.zeros((num_tasks, num_tasks))
    memory, fisher, star, teacher = [], None, None, None

    for t in range(num_tasks):
        log(f"\n  --- Task {t+1}/{num_tasks} ({method}) ---")
        use_mem = memory if method in ("replay", "combined") else None
        use_fisher = fisher if method in ("ewc", "combined") else None
        use_star = star if method in ("ewc", "combined") else None
        use_lambda = ewc_lambda if method in ("ewc", "combined") else 0.0
        use_teacher = teacher if method == "lwf" else None

        train_task(model, task_train[t], ITERS_PER_TASK, LEARNING_RATE,
                   BATCH_SIZE, memory=use_mem, fisher=use_fisher,
                   star=use_star, ewc_lambda=use_lambda, teacher=use_teacher)

        # update EWC state
        if method in ("ewc", "combined"):
            fisher = estimate_fisher(model, task_train[t], FISHER_SAMPLES)
            star = {n: p.detach().clone() for n, p in named_trainable(model)}
        # update LwF teacher
        if method == "lwf":
            teacher = copy.deepcopy(model).eval()
            for p in teacher.parameters():
                p.requires_grad = False
        # update replay buffer (Mod 4)
        if method in ("replay", "combined"):
            ds = task_train[t]
            paths = [ds.get_path_and_label(i)[0] for i in range(len(ds))] \
                if hasattr(ds, "get_path_and_label") else ds.image_paths
            labels = [ds.get_path_and_label(i)[1] for i in range(len(ds))] \
                if hasattr(ds, "get_path_and_label") else ds.labels
            wt = getattr(ds, "window_type", None)
            memory += select_exemplars(model, paths, labels, wt, BUFFER_PER_TASK)
            log(f"    buffer size: {len(memory)}")

        # evaluate on all tasks seen so far
        for j in range(num_tasks):
            agg = patient_of is not None
            sl, pat = evaluate(model, task_test[j],
                               paths=test_paths[j] if test_paths else None,
                               patient_of=patient_of, aggregate=agg)
            R[t, j] = sl
            if pat is not None:
                Rpat[t, j] = pat
        log(f"    R[{t+1}] = {np.round(R[t], 2).tolist()}")
    return R, Rpat


def avg_acc(R):
    return float(R[-1].mean())


def forgetting(R):
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[:T-1, j].max() - R[T-1, j] for j in range(T-1)]))


# ============================================================================
# MAIN
# ============================================================================
def main():
    set_seed(42)
    log("=" * 70)
    log("IMPROVED EXPERIMENT (mods 1-6)")
    log(f"Device: {DEVICE} | DEBUG_RUN={DEBUG_RUN}")
    log(f"Backbone: pretrained ResNet-18 | input {TARGET_SIZE}")
    log(f"EWC lambda grid: {EWC_LAMBDA_GRID} | buffer/task: {BUFFER_PER_TASK} "
        f"| herding: {HERDING} | LwF(alpha={LWF_ALPHA}, T={LWF_TEMP})")
    log("=" * 70)

    train_paths, train_labels, test_paths, test_labels = load_rsna_data(
        DATA_ROOT, LABELS_CSV, test_split=0.2,
        max_patients_per_class=MAX_PATIENTS,
        max_images_per_patient=MAX_IMAGES, seed=42,
    )

    def patient_of(p):
        # data/RSNA2023ProcessedImages/<patient_id>/<series>/<file>.png
        return Path(p).parent.parent.name

    # Class-incremental tasks (the headline setting)
    train_base = RSNADataset(train_paths, train_labels, TARGET_SIZE)
    test_base = RSNADataset(test_paths, test_labels, TARGET_SIZE)
    num_tasks = 2
    task_train = create_task_datasets(train_base, num_tasks)
    task_test = create_task_datasets(test_base, num_tasks)
    test_paths_per_task = [
        [task_test[j].get_path_and_label(i)[0] for i in range(len(task_test[j]))]
        for j in range(num_tasks)
    ]

    results = {}
    # Methods that don't depend on lambda use the first grid value as a no-op.
    plan = [("baseline", None), ("replay", None), ("lwf", None)]
    for lam in EWC_LAMBDA_GRID:
        plan.append((f"ewc(λ={lam:g})", lam))
        plan.append((f"combined(λ={lam:g})", lam))

    for name, lam in plan:
        base_method = name.split("(")[0]
        lam_val = lam if lam is not None else EWC_LAMBDA_GRID[0]
        log(f"\n{'='*70}\nMETHOD: {name}\n{'='*70}")
        R, Rpat = run_method(base_method, task_train, task_test, num_tasks,
                             lam_val, test_paths=test_paths_per_task,
                             patient_of=patient_of)
        results[name] = {
            "R": R, "Rpat": Rpat,
            "avg_acc": avg_acc(R), "forgetting": forgetting(R),
            "avg_acc_patient": avg_acc(Rpat), "forgetting_patient": forgetting(Rpat),
        }
        log(f"  {name}: slice ACC={results[name]['avg_acc']:.2f} "
            f"FGT={results[name]['forgetting']:.2f} | "
            f"patient ACC={results[name]['avg_acc_patient']:.2f}")

    # Save results
    out_csv = LOG_DIR / f"results_improved_{TIMESTAMP}.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Method,Avg_Accuracy,Forgetting,Avg_Accuracy_Patient,Forgetting_Patient\n")
        for name, r in results.items():
            f.write(f"{name},{r['avg_acc']:.4f},{r['forgetting']:.4f},"
                    f"{r['avg_acc_patient']:.4f},{r['forgetting_patient']:.4f}\n")
    np.savez(LOG_DIR / f"matrices_improved_{TIMESTAMP}.npz",
             **{k.replace("(", "_").replace(")", "").replace("=", "")
                 .replace("λ", "lam").replace(".", "p"): v["R"]
                for k, v in results.items()})

    log(f"\n{'='*70}\nSUMMARY (improved)\n{'='*70}")
    log(f"{'Method':<22}{'Slice ACC':>12}{'FGT':>10}{'Patient ACC':>14}")
    for name, r in results.items():
        log(f"{name:<22}{r['avg_acc']:>11.2f}%{r['forgetting']:>9.2f}%"
            f"{r['avg_acc_patient']:>13.2f}%")
    log(f"\nResults: {out_csv}")
    log("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"Total time: {time.time()-t0:.1f}s")
