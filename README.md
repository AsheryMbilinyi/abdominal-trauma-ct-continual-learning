# What Limits Continual Learning for Abdominal-Trauma CT Detection?

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-009E73)
![Topic](https://img.shields.io/badge/Topic-Continual%20Learning-CC79A7)
![Methods](https://img.shields.io/badge/Methods-EWC%20%7C%20Replay%20%7C%20LwF%20%7C%20MIL-0072B2)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Code for the MICAD 2026 paper *"What Limits Continual Learning for
Abdominal-Trauma CT Detection? A Diagnostic Study of Forgetting vs.
Representation Bottlenecks."* We run a three-stage diagnostic sequence
(custom CNN → pretrained ResNet-18 → patient-level attention-MIL) comparing
**fine-tuning**, **EWC**, **Experience Replay**, **EWC+Replay**, and **LwF**
on the RSNA 2023 Abdominal Trauma Detection dataset, under a class-incremental
stream (Experiment 1) and a window-based domain-incremental stream
(Experiment 2).

**Tags:** `continual-learning` · `catastrophic-forgetting` · `ewc` ·
`experience-replay` · `learning-without-forgetting` · `multiple-instance-learning` ·
`medical-imaging` · `ct` · `rsna` · `pytorch`

## Repository layout

Folders mirror the paper's stage/experiment structure (Table 2) directly, so
each script's location tells you where it's reported.

```
.
├── src/
│   ├── stage1_cnn/                    # Stage I: custom CNN (Sec. 3.2, Sec. 5.1)
│   │   ├── exp1_class_incremental.py      # Exp 1: 2-task class-incremental (headline)
│   │   └── exp2_window_domain.py          # Exp 2: 3-task window domain-incremental
│   ├── stage2_resnet18/               # Stage II: pretrained ResNet-18 (Sec. 5.2)
│   │   └── exp1_ewc_sweep_lwf.py          # Exp 1 only — EWC λ-sweep, LwF added
│   ├── stage3_mil/                    # Stage III: patient-level attention-MIL (Sec. 5.3)
│   │   ├── exp2_attention_mil.py          # Exp 2 only — Run A / Run B in the paper
│   │   └── tune_mil.py                    # validation-AUC sweep feeding Run B config
│   ├── archive/                       # exploratory runs, NOT reported in the paper
│   │   ├── exp_window_3task_v3.py         # later 3-window variant (near-chance)
│   │   └── exp_improved_v2.py             # ResNet-18 slice-level 3-window run,
│   │                                       #   superseded by Stage III's MIL approach
│   ├── config.py                      # central experiment configuration + presets
│   ├── utils.py                       # helpers
│   └── quickstart.py                  # scaffold (does not load data)
├── notebooks/                     # interactive walkthroughs (local only; empty on GitHub)
├── report/                        # report
│   ├── figures/                       # committed vector figures (PDF)
│   └── legacy/                        # earlier plain-text reports (local only; empty on GitHub)
├── logs/                          # run logs / CSVs / matrices (local only; empty on GitHub)
└── data/                          # dataset (local only; empty on GitHub)
```

> **Note on `archive/`:** these two scripts were exploratory steps that aren't
> part of the diagnostic sequence reported in the paper. Stage II is evaluated
> only on Experiment 1 (Table 2), and `exp_improved_v2.py`'s slice-level
> 3-window attempt is superseded by Stage III's patient-level MIL
> reformulation, which is what the paper reports for Experiment 2 beyond
> Stage I.

## Data layout

Expected on disk under `data/`:

```
data/
  RSNA2023ProcessedImages/<patient_id>/<series_id>/<instance>.png
  train.csv               # labels; uses the `any_injury` column
  image_level_labels.csv
```

## Running the experiments

```powershell
conda activate medical_ml

# Stage I, Experiment 1 — smoke test then full run
$env:DEBUG_RUN = "1"; python src/stage1_cnn/exp1_class_incremental.py
Remove-Item Env:DEBUG_RUN -ErrorAction SilentlyContinue
python src/stage1_cnn/exp1_class_incremental.py

# Stage I, Experiment 2
python src/stage1_cnn/exp2_window_domain.py
```

### Stage II: pretrained ResNet-18 (Experiment 1 only)

`src/stage2_resnet18/exp1_ewc_sweep_lwf.py` implements the six modifications
derived from the Stage I diagnosis: (1) pretrained ResNet-18 backbone, (2) EWC
λ sweep {10, 50, 100, 500, 1000}, (3) balanced replay loss, (4) larger buffer
+ herding exemplar selection, (5) LwF knowledge distillation, (6)
patient/series-level label aggregation. Self-contained (no full-run side
effects on import).

```powershell
$env:DEBUG_RUN = "1"; python src/stage2_resnet18/exp1_ewc_sweep_lwf.py     # fast smoke test
Remove-Item Env:DEBUG_RUN; python src/stage2_resnet18/exp1_ewc_sweep_lwf.py # full run (multi-hour, GPU)
```

### Stage III: patient-level attention-MIL (Experiment 2 only)

`src/stage3_mil/exp2_attention_mil.py` pools `K=40` slices per patient via
gated attention (Eq. 2 in the paper) and matches the bag label to
`any_injury` exactly, fixing the slice-level label mismatch left uncorrected
through Stages I–II. `tune_mil.py` runs the validation-AUC hyperparameter
sweep that selects the tuned configuration used for the paper's Run B.

```powershell
python src/stage3_mil/tune_mil.py            # hyperparameter sweep (Run A config search)
python src/stage3_mil/exp2_attention_mil.py  # full run with tuned config (Run B)
```

## Citation

If you use this code, please cite the paper (MICAD 2026):

```bibtex
@inproceedings{ismaila2026abdominal,
  title     = {What Limits Continual Learning for Abdominal-Trauma CT Detection?
               A Diagnostic Study of Forgetting vs. Representation Bottlenecks},
  author    = {Ismaila, Lukman E. and Roy, Nidita and Amin, Smit and
               Maroa, Cornelius and Mbilinyi, Ashery},
  booktitle = {Medical Imaging and Computer-Aided Diagnosis (MICAD)},
  year      = {2026}
}
```
