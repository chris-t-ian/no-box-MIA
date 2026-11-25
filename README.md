# No-Box Membership Inference Attacks on 3D Medical Generative Models

Code accompanying the MSc thesis of Christian Schneider on **membership inference against synthetic 3D brain MRI**. The repository studies a *no-box* threat model: the adversary never queries the generative model and has no access to its weights or losses — they only see the **released synthetic dataset** plus some auxiliary real data from the same distribution.

The main attack is a density-ratio attack in the spirit of **DOMIAS** (van Breugel et al., 2023), using **Neural Spline Flows** (NSF) as density estimators on volumetric data. The target model is a **VQ-VAE + autoregressive Transformer** trained on T1-weighted MRI from **ATLAS v2.0**.

## Research questions covered by the code

- **3D → 3D attack:** Can a DOMIAS adversary infer membership from 3D synthetic volumes?
- **3D → 2D attack:** Does the attack still work if the adversary only models 2D middle slices (axial) of real and synthetic volumes? This makes density estimation much cheaper.
- **Adversary background knowledge:** How does performance change when the adversary's auxiliary set overlaps 100 % / 50 % / 0 % with the target model's training set?
- **Outliers:** Are low/high-density targets more vulnerable? (AUC and TPR@low-FPR on percentile subsets of the targets.)
- **Density-estimator overfitting:** Does overfitting the raw-data NSF (fixed 200/400 epochs vs. early stopping) change attack performance?
- **Baselines and variants (experimental):** LOGAN-style (`p_S` only), z-calibrated DOMIAS, and LiRA-style shadow-model attacks.

## Threat model

The experiments are set up as a membership game between two parties (`p_vqvae/mia.py`):

**`Challenger`** (data holder / model trainer)

1. Samples `n_c` training subjects out of the full ATLAS distribution (955 subjects) using a fixed data seed.
2. Trains the target generator (VQ-VAE, then a Transformer prior over the VQ codes).
3. Releases `m_c` synthetic samples.
4. Draws targets with membership labels `b ∈ {0, 1}`. By default every subject is a target.

**`AdversaryDOMIAS`**

1. Gets an auxiliary real dataset of size `n_a`. Its overlap with the challenger's training set is set by `background_knowledge ∈ {1.0, 0.5, 0.0}` (or any fraction).
2. Fits one NSF on the synthetic data (`p_S`) and one NSF on the auxiliary real data (`p_R`).
3. Scores each target `x` by `log p_S(x) − log p_R(x)`, normalizes the score to [0, 1] and sweeps a threshold to get ROC, AUC and TPR at FPR = 0.001.
4. In the 3D → 2D setting, the axial middle slice of both the synthetic volumes and the real volumes is used instead (`data_type_adversary="2D_MRI"`).

## Repository structure

```
.
├── main.py                          # Entry point: chooses which experiments to run (thesis figures)
├── p_vqvae/
│   ├── experiments.py               # All experiment definitions and global configuration (hyperparameters, paths, device)
│   ├── mia.py                       # Challenger, AdversaryDOMIAS, LOGAN, z-calibrated DOMIAS, LiRA variants
│   ├── networks.py                  # VQ-VAE, Transformer decoder (MONAI), membership classifier, BaseModel (save/load)
│   ├── neural_spline_flow.py        # 2D/3D multi-scale NSF (nflows): squeeze, ActNorm, 1x1 conv, RQ-spline couplings
│   ├── dataloader.py                # ATLAS 3D/2D, sklearn digits, synthetic datasets, augmentation, loaders
│   ├── utils.py                     # Slicing, downsampling, AUC / TPR@FPR helpers, subset hashing
│   ├── visualise.py                 # ROC curves, AUC plots, histograms, sample plots
│   ├── metrics.py                   # FID (2D), Fréchet MedicalNet distance (3D), MS-SSIM
│   ├── hyperparameter_search_nsf.py # Optuna search for NSF
│   └── hyperparameter_search_vqvae.py
└── tests/
    └── test_squeeze_transform.py    # Unit tests for the 3D squeeze transform
```

## Installation

The repository has no `requirements.txt`. These are the packages it imports:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch monai nflows nibabel numpy scipy pandas scikit-learn matplotlib tqdm optuna pytest
# only needed for p_vqvae/metrics.py:
pip install torchmetrics pytorch-msssim
```

Notes:

- The VQ-VAE, `DecoderOnlyTransformer` and `VQVAETransformerInferer` are imported from `monai.networks.nets` / `monai.inferers`. This requires a MONAI version that includes the former MONAI Generative models (**MONAI ≥ 1.4**, likely).
- `metrics.py` needs [MedicalNet](https://github.com/Tencent/MedicalNet) on the Python path (`from MedicalNet.model import ...`).
- `experiments.py` has the line `from PrivateVAE.p_vqvae.dataloader import ...`. This only resolves inside the author's original directory layout. Run from the parent directory of a checkout named `PrivateVAE`, or change the import to `from p_vqvae.dataloader import ...`.
- A CUDA GPU is effectively required. The device is hard-coded as `device = "cuda:5"` at the top of `experiments.py`; change it to match your machine.

## Data

### ATLAS v2.0 (main dataset)

[Anatomical Tracings of Lesions After Stroke (ATLAS) R2.0](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html). Access requires a data use agreement. The loader walks `data/ATLAS_2/` recursively and picks up every `*T1w.nii.gz` (955 subjects in the training release).

Preprocessing (`raw_data_kwargs` in `experiments.py`):

| Step | Setting |
|---|---|
| Crop | `((8, 9), (12, 13), (0, 9))` voxels |
| Downsample | factor 4 (strided: every 4th voxel) |
| Normalize | min–max to [0, 1] per volume |
| Pad | `((1, 2), (0, 0), (1, 2))` |

With the standard 197 × 233 × 189 ATLAS volumes, this gives volumes of about **48 × 52 × 48**. These sizes are divisible by 4, which the two-level squeeze in the NSF needs.

On first load the preprocessed tensor is cached in `data/cache/ATLAS_2_full_smpl4.npy`. 2D slices are cached alongside it as `..._2D_axial_ds1.npy`. Later runs memory-map these files.

### sklearn digits (sanity check)

`data_type="digits"` uses `sklearn.datasets.load_digits()` (8×8 images). Nothing needs to be downloaded.

### Expected directory layout

Create these folders before running (plots are written with `plt.savefig` and are not created automatically):

```
data/
├── ATLAS_2/        # raw NIfTI files
├── cache/          # preprocessed .npy cache (auto-filled)
├── plots/          # figures
├── results/        # LaTeX tables of results
└── logs/
model_outputs/
└── mia/            # trained VQ-VAE / Transformer / NSF weights + synthetic_data/
```

```bash
mkdir -p data/{ATLAS_2,cache,plots,results,logs} model_outputs/mia
```

## Usage

All experiments are plain functions in `p_vqvae/experiments.py`. `main.py` calls a selection of them; comment lines in or out to choose what runs:

```bash
python main.py
```

| Function | What it does | Thesis |
|---|---|---|
| `domias(data_type_challenger="3D_MRI", data_type_adversary="3D_MRI", plot_roc=True)` | 3D DOMIAS attack on 3D synthetic data | Fig. 6 |
| `domias(data_type_challenger="3D_MRI", data_type_adversary="2D_MRI", plot_roc=True)` | 2D (axial middle slice) attack on 3D synthetic data | Fig. 6 |
| `domias_vary_knowledge(n_a_mode="n_c" \| "all", ...)` | Background knowledge 1.0 / 0.5 / 0.0. `n_a = n_c` or `n_a = 955 − n_c` | Fig. 7 |
| `domias_outlier_3d_to_2d(n_target_seeds, adjust_n_targets)` | AUC / TPR@FPR on target percentiles (1.0, .75, .5, .25, .1) over several target draws | Fig. 5 (?) |
| `domias_exagerate_nsf_overfitting(...)` | Raw-data NSF trained with early stopping vs. 200 / 400 epochs (knowledge = 0) | – |
| `show_nsf_samples(...)` | Samples from the raw and synthetic NSFs (sanity check) | – |
| `plot_learning_curve_vqvae_and_transformer()`, `plot_learning_curves_nsf()` | Training curves | – |
| `domias_multiseed`, `domias_overfitting`, `domias_downsampling_2d`, `logan`, `zdomias_*`, `LIRA_classifier` | Additional or exploratory experiments; some depend on `challenger_standard_mri`, which is set to `None` | – |

The figure numbers come from comments in `main.py`.

Minimal programmatic example:

```python
from p_vqvae.experiments import domias

res = domias(
    data_type_challenger="3D_MRI",   # target generator trained on 3D volumes
    data_type_adversary="2D_MRI",    # adversary works on axial middle slices
    challenger_seed=0,
    adversary_knowledge=1.0,         # auxiliary set = challenger training set
    plot_roc=True,
)
print(res["auc"], res["tpr_at_low_fpr"])
```

### Key configuration (top of `experiments.py`)

| Variable | Default | Meaning |
|---|---|---|
| `challenger_kwargs_mri` | `n_c=200`, `m_c=1000`, `n_targets=955` | Training-set size, number of synthetic samples, number of targets |
| `vqvae_kwargs` | 200 epochs, 512 codes × 64 dim, channels (384, 512) | Target model stage 1 |
| `transformer_kwargs` | 200 epochs, 12 layers, 12 heads, dim 96 | Target model stage 2 |
| `nsf_kwargs_mri` | 2 levels × 10 steps, RQ-spline (4 bins), ActNorm, 1×1 conv, lr 4e-4, ≤400 epochs, early stopping 5 | Density estimators of the adversary |
| `low_fpr` | 0.001 | FPR used for TPR@low-FPR |

### Caching and reproducibility

- Model weights are saved and loaded under `model_outputs/mia/`. The file names encode the relevant hyperparameters and a SHA-256 hash of the training-ID subset (`BaseModel.get_full_path`). Re-running an experiment with identical settings reuses trained models. **Delete the folder after changing anything the file name does not encode** (e.g. augmentation, architecture details).
- Synthetic datasets are stored as `model_outputs/mia/synthetic_data/chall_syn_*.npy`.
- The challenger's training subset and the adversary's auxiliary subset are seeded with `challenger_seed`. Target sampling (`Challenger.sample_n_random_target_ids`) uses a fresh `random.seed()` and is **not** deterministic.

## Tests

```bash
pytest tests/
```

Only `SqueezeTransform` (shape checks and invertibility constraints) is covered.

## Known limitations

Based on a code review. None of these have been changed in the repository.

- **Portability:** hard-coded GPU (`cuda:5`), absolute paths in the hyperparameter-search scripts (`/home/chrsch/...`), the `PrivateVAE.` import, no pinned dependencies, no license file.
- **Score direction:** in `AdversaryDOMIAS.infer_membership_logx`, a target is predicted to be a member when the normalized `log p_S − log p_R` is **below** the threshold. DOMIAS and `MultiSeedAdversaryDOMIAS` (which uses `sklearn.metrics.roc_curve` on the same score) treat a **high** ratio as evidence for membership. The single-seed and multi-seed paths therefore likely rank targets in opposite directions. Check this before comparing AUCs across experiments.
- **Outlier selection:** `select_outliers` keeps the targets with the **highest** `log p_R`, while its docstring says it selects low-density targets. Check which subset the outlier experiments were meant to use.
- **Early stopping in NSF:** `best_model_weights = self.model.state_dict()` stores a reference, not a copy. "Restoring the best weights" therefore keeps the last weights.
- **Target-model validation set:** the first 20 % of non-members is used for early stopping of the target model. These subjects are also non-member targets.
- `NSF.max_num_steps = epochs * batch_size` is only used for the cosine-annealing schedule, which is off by default.
- LiRA, z-calibrated DOMIAS and RMIA classes are partly unfinished (`NotImplementedError`, `TODO`s, references to `challenger_standard_mri = None`).

## References

- B. van Breugel, H. Sun, Z. Qian, M. van der Schaar. *Membership Inference Attacks against Synthetic Data through Overfitting Detection (DOMIAS).* AISTATS 2023.
- C. Durkan, A. Bekasov, I. Murray, G. Papamakarios. *Neural Spline Flows.* NeurIPS 2019.
- N. Carlini et al. *Membership Inference Attacks From First Principles (LiRA).* IEEE S&P 2022.
- J. Hayes, L. Melis, G. Danezis, E. De Cristofaro. *LOGAN: Membership Inference Attacks Against Generative Models.* PoPETs 2019.
- P.-D. Tudosiu et al. *Morphology-Preserving Autoregressive 3D Generative Modelling of the Brain.* SASHIMI @ MICCAI 2022.
- S.-L. Liew et al. *A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms (ATLAS v2.0).* Scientific Data, 2022.

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{schneider_nobox_mia,
  author = {Christian Schneider},
  title  = {<thesis title>},
  school = {<university>},
  year   = {2025}
}
```
