# PoseJEPA

### Pose-Conditioned Latent Prediction for Unseen Object Pose Estimation

**Master's thesis project** · Laurea Magistrale in Computer Science and Engineering  
**Politecnico di Milano** · A.Y. 2025–2026

**Author:** Riccardo Portaro  
**Advisor:** Federica Arrigoni · **Co-advisor:** Angelo Iacoella

PoseJEPA investigates latent-space prediction for model-based 6D pose estimation of unseen objects. Given an RGB-D image, an object mask, a CAD model, and camera intrinsics, the method estimates the object's pose without having seen that object during training.

> **Project status:** This repository contains research code for the thesis. The experiment configuration uses machine-specific data and output paths; adapt them to your environment before running the pipeline.

## Overview

Many state-of-the-art pose-estimation methods use a *render-and-compare* loop: render an object at a candidate pose, compare the rendered result with the observation, and refine the pose. Repeated rendering can be expensive, and pixel-level differences may be sensitive to appearance and image noise.

PoseJEPA instead predicts the expected image features directly in latent space. A pose-conditioned predictor produces DINOv3 features for a pose hypothesis, and the discrepancy with features extracted from the observed image defines an energy:

```text
T* = argmin_{T in SE(3)} D(Phi(M; T), Psi(I))
```

Here, `M` is the CAD model, `I` is the observed image, `Phi` is the pose-conditioned latent predictor, `Psi` is the target-image encoder, and `D` is a feature discrepancy.

## Method

### Training

PoseJEPA has two feature-processing branches:

1. **Target encoder:** a frozen DINOv3 encoder extracts intermediate-layer image features. Freezing the encoder preserves its pretrained representation and avoids representation collapse.
2. **Multi-view 3D representation:** points sampled from a CAD mesh are associated with DINOv3 features back-projected from multiple rendered viewpoints, along with their view directions.
3. **Aggregator:** a transformer decoder queries the multi-view representation at each 3D point and combines the point's features and view directions into an aggregated feature.
4. **Pose-conditioned predictor:** a transformer predicts image features from the aggregated object representation. Plücker ray coordinates condition the prediction on the target pose, without volumetric rendering.

The predictor is trained with a masked L1 loss over the visible object region.

### Inference

1. **Coarse pose hypotheses:** sample rotations uniformly on SO(3) using Fibonacci sampling and initialize translation from a geometric prior.
2. **Pose refinement:** iteratively refine each hypothesis with a learned refiner that compares predicted and target latent representations.
3. **Hypothesis selection:** select the refined pose with the lowest latent-space energy.

The object representation is precomputed and cached once per object; the reported preprocessing time is under 15 seconds per object.

## Results

Evaluated in the localization setting on five BOP datasets using the default BOP Average Recall (AR) protocol:

| Method | Segmentation | Refinement | LM-O | YCB-V | IC-BIN | TUD-L | T-LESS | Mean AR |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| GigaPose | CNOS | ✓ | 0.678 | 0.825 | 0.563 | 0.811 | 0.556 | 0.687 |
| MegaPose | CNOS | ✓ | 0.626 | 0.764 | 0.467 | 0.851 | 0.487 | 0.639 |
| FoundationPose | CNOS | ✓ | 0.756 | 0.889 | 0.508 | 0.923 | 0.646 | 0.744 |
| FreeZeV2 | CNOS | ✓ | 0.764 | 0.906 | 0.654 | 0.972 | 0.708 | 0.801 |
| **PoseJEPA (ours)** | CNOS | ✓ | 0.483 | 0.642 | 0.380 | 0.473 | 0.488 | 0.493 |
| **PoseJEPA (ours)** | Ground-truth masks | ✓ | 0.612 | 0.747 | 0.562 | 0.716 | 0.692 | 0.667 |

PoseJEPA does not reach state-of-the-art performance, but it is trained on approximately 1,000 objects—up to 40 times fewer than other approaches in this comparison. Refinement consistently improves its results. Segmentation quality also has a substantial effect: mean AR is 0.493 with CNOS masks and 0.667 with ground-truth masks.

## Limitations and future work

- **The energy is not fully pose-aware:** masked L1 aligns predictions with DINOv3 features but does not directly optimize geometric pose proximity.
- **Synthetic-to-real domain gap:** this gap in DINO representations is made harder by regressing features rather than matching them.
- **Translation ambiguity:** estimating translation, particularly from depth, remains challenging.

Future work includes more pose-aware representations and energy landscapes, as well as methods to reduce the synthetic-to-real gap.

## Repository structure

```text
.
├── configs/                  # Hydra configuration
├── Essential/                # Dataset, output, render, and detection paths
├── Script/                   # Dataset utilities and representation precomputation
├── src/
│   ├── data/                 # Datasets, streams, and rendering utilities
│   ├── models/
│   │   ├── predictor.py      # Pose-conditioned predictors
│   │   ├── encoder2d/        # DINOv2 and DINOv3 encoders
│   │   ├── encoder3d/        # 3D encoders and aggregator
│   │   └── crosspredictor.py # Cross-attention predictor
│   ├── refinement.py         # Pose refinement
│   ├── refinement2.py        # Pose refinement
│   └── engine*.py            # Training engines
├── Qualitative/              # Qualitative analysis and visualization
├── bop_toolkit/              # BOP evaluation toolkit
├── train.py                  # Hydra-configured training entry point
└── eval.py                   # BOP evaluation wrapper
```

## Setup and usage

There is currently no root-level `requirements.txt`, `environment.yml`, or `pyproject.toml` defining a reproducible environment. The project uses PyTorch, Hydra, and GPU-based rendering components; install compatible dependencies for your system before running the scripts.

Clone the repository:

```bash
git clone https://github.com/Saiukk/PoseJEPA.git
cd PoseJEPA
```

Before running an experiment:

- Configure dataset, model, cache, and output locations in [`configs/`](configs/) and [`Essential/path.py`](Essential/path.py). Some current configuration and script paths are specific to the original machine.
- Prepare the required BOP datasets and 3D models. The default training configuration also reads the MegaPose-GSO shards from Hugging Face.
- Configure access to the DINOv3 model specified in [`configs/encoder2d/dinov3.yaml`](configs/encoder2d/dinov3.yaml), if required by your Hugging Face setup.
- Check GPU and rendering-library compatibility for your environment.

### Train

The training entry point uses Hydra and the default configuration in `configs/config.yaml`:

```bash
python train.py
```

Hydra configuration values can be overridden on the command line, for example:

```bash
python train.py trainer.epochs=100
```

### Precompute object representations

For BOP CAD models, `Script/precompute_BOP.py` accepts the model and output directories as Hydra overrides:

```bash
python Script/precompute_BOP.py \
  +bop.models_dir=/path/to/bop/models \
  +bop.out_dir=/path/to/representation-cache
```

The preprocessing script uses the encoder and representation settings from the project configuration. Ensure these match the settings used for training.

### Evaluate

`eval.py` is currently an experiment-specific BOP evaluation wrapper, not a general command-line interface: it contains fixed dataset, result, and toolkit paths. Update those paths and the result filename in the script for your experiment before running:

```bash
python eval.py
```

## Citation

```bibtex
@mastersthesis{portaro2026posejepa,
  title  = {PoseJEPA: Exploring Pose-Conditioned Latent Prediction for Unseen Object Pose Estimation},
  author = {Portaro, Riccardo},
  school = {Politecnico di Milano},
  year   = {2026}
}
```

## Acknowledgments

Thanks to my advisor Federica Arrigoni and co-advisor Angelo Iacoella. This work builds on ideas from [I-JEPA](https://arxiv.org/abs/2301.08243), [DINOv3](https://arxiv.org/abs/2508.10104), [FoundationPose](https://arxiv.org/abs/2312.08344), and the [BOP benchmark](https://bop.felk.cvut.cz/).

## License

No license has been specified in this repository yet. All rights reserved unless the author states otherwise.
