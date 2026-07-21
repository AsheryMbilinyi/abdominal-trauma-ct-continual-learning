# ==============================================================================
# Targeted hyperparameter sweep for the patient-level Attention-MIL model.
# ==============================================================================
# Optimises VALIDATION patient AUC only. The test set is never touched here, so
# the final reported numbers (from exp2_attention_mil.py) stay honest. Each configuration
# trains a single cheap proxy: the BASELINE method on ONE task (Brain window)
# for a reduced number of epochs. The configuration with the best validation
# AUC is printed at the end; we then plug it into exp2_attention_mil.py for the full run.
#
# Usage:
#   conda activate medical_ml
#   python src/tune_mil.py            # full sweep on the real (sub)set
#   $env:DEBUG_RUN="1"; python src/tune_mil.py   # tiny smoke sweep
# ==============================================================================

import os
import time
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

import exp2_attention_mil as M   # safe to import: exp2_attention_mil guards main() under __main__

DEBUG = os.environ.get("DEBUG_RUN", "0") == "1"

# Proxy training is cheaper than the full run.
PROXY_EPOCHS = 1 if DEBUG else 6
PROXY_PATIENTS_PER_CLASS = 8 if DEBUG else 400   # enough to rank configs, ~half-data
PROXY_TASK = 0                                    # Brain window only

# Search grid (high-impact knobs only).
GRID = {
    "ENCODER":            (["resnet18"] if DEBUG else ["resnet18", "resnet34"]),
    "ENCODER_LR":         ([1e-4] if DEBUG else [3e-5, 1e-4, 3e-4]),
    "MAX_SLICES_PER_BAG": ([4] if DEBUG else [24, 40]),
    "ATT_DIM":            ([128] if DEBUG else [128, 256]),
}

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = LOG_DIR / f"tune_mil_{TS}.csv"


def proxy_score(cfg, train_bags, val_bags):
    """Train baseline on one task with this config; return best val AUC."""
    # apply config to exp_mil globals
    M.ENCODER = cfg["ENCODER"]
    M.ENCODER_LR = cfg["ENCODER_LR"]
    M.MAX_SLICES_PER_BAG = cfg["MAX_SLICES_PER_BAG"]
    M.EPOCHS_PER_TASK = PROXY_EPOCHS
    M.EARLY_STOP_PATIENCE = 99   # no early stop during short proxy

    M.set_seed(42)
    model = M.AttentionMIL(att_dim=cfg["ATT_DIM"]).to(M.DEVICE)

    # train_task evaluates val AUC each epoch and keeps the best-AUC weights;
    # we recover that best by evaluating the returned (best-state) model.
    model = M.train_task(model, train_bags, val_bags, PROXY_TASK, tag="proxy")
    val_acc, val_auc = M.evaluate(model, val_bags, PROXY_TASK)
    del model
    if M.DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return val_acc, val_auc


def main():
    M.log("=" * 72)
    M.log(f"MIL HYPERPARAMETER SWEEP (proxy: baseline, Brain window, {PROXY_EPOCHS} epochs)")
    M.log(f"Optimising VALIDATION patient AUC | DEBUG={DEBUG}")
    M.log("=" * 72)

    # smaller patient cap for fast, fair ranking
    M.MAX_PATIENTS_PER_CLASS = PROXY_PATIENTS_PER_CLASS
    train_bags, val_bags, _ = M.load_bags()

    keys = list(GRID.keys())
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
    M.log(f"Configurations to try: {len(configs)}")

    rows = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        M.log(f"\n[{i+1}/{len(configs)}] {cfg}")
        try:
            acc, auc = proxy_score(cfg, train_bags, val_bags)
        except RuntimeError as e:
            M.log(f"   FAILED: {e}")
            acc, auc = float("nan"), float("nan")
        dt = time.time() - t0
        M.log(f"   -> val acc {acc:.1f}% | val AUC {auc:.1f} | {dt:.0f}s")
        rows.append((cfg, acc, auc))

    rows_valid = [r for r in rows if not np.isnan(r[2])]
    rows_valid.sort(key=lambda r: r[2], reverse=True)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("rank,encoder,encoder_lr,slices_per_bag,att_dim,val_acc,val_auc\n")
        for rk, (cfg, acc, auc) in enumerate(rows_valid, 1):
            f.write(f"{rk},{cfg['ENCODER']},{cfg['ENCODER_LR']},"
                    f"{cfg['MAX_SLICES_PER_BAG']},{cfg['ATT_DIM']},{acc:.4f},{auc:.4f}\n")

    M.log(f"\n{'='*72}\nRANKED CONFIGS (by validation AUC)\n{'='*72}")
    for rk, (cfg, acc, auc) in enumerate(rows_valid, 1):
        M.log(f"  {rk}. AUC {auc:5.1f} acc {acc:5.1f}% | {cfg}")
    if rows_valid:
        best = rows_valid[0][0]
        M.log(f"\nBEST: {best}")
        M.log("Set these in exp2_attention_mil.py for the final full run.")
    M.log(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
