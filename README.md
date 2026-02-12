# LenslessInfoDesign: Designing lensless imaging systems to maximize information capture

[![paper-Optica](https://img.shields.io/badge/paper-Optica-5C2D91.svg)](https://doi.org/10.1364/OPTICA.570334)
[![arXiv](https://img.shields.io/badge/arXiv-2506.08513-b31b1b.svg)](https://arxiv.org/abs/2506.08513)




## About

This repository contains code for the paper "Designing lensless imaging systems to maximize information capture."

All Jupyter notebooks can be viewed without execution if the reader cannot or does not want to run them.

## Installation

Please recursively clone this Git repository to include the EncodingInformation submodule:

```bash
git clone --recurse-submodules https://github.com/lakabuli/LenslessInfoDesign.git
```

Alternatively, EncodingInformation can be manually cloned from https://github.com/Waller-Lab/EncodingInformation.

See the [EncodingInformation Installation Guide](https://github.com/Waller-Lab/EncodingInformation/blob/main/Installation_guide.md) for environment setup instructions via pip.

**System Requirements:** All experiments were run on a Linux server with a single Nvidia RTX A6000 GPU.

## Repository Structure

```
LenslessInfoDesign/
├── design_IDEAL/              # Sec. 4: Information-Optimal Encoder Design
│   ├── optimization_scripts/   # PSF optimization scripts
│   ├── extended_fov_scripts/   # Extended field-of-view reconstruction
│   └── data/                   # Optimized PSFs, models, and estimates
├── experimental_eval/         # Sec. 5: Experimental Information Evaluation
│   └── data/                   # Reconstructions and estimates
├── tradeoff_analysis/         # Sec. 3: Quantifying Sparsity and Multiplexing Tradeoffs
│   ├── mi_sweep_scripts/      # Scripts for mutual information estimation sweeps
│   ├── tc_sweep_scripts/      # Scripts for Tamura coefficient sweeps
│   ├── mi_estimates/           # Mutual information estimates
│   └── tc_values/              # Tamura coefficient values
└── figures/                   # Generated figure components
```

## Model Availability

Sec. 5 (Experimental Information Evaluation) uses a data-driven reconstruction algorithm (ConvNeXt) from Ponomarenko et al. [1], generously provided by Vasilisa Ponomarenko.

**[1]** V. Ponomarenko, L. Kabuli, E. Markley, C. Hung, L. Waller, "Phase-mask-based lensless image reconstruction using self-attention," Proc. SPIE PC13333, Paper 12.3042497 (2024). [https://doi.org/10.1117/12.3042497](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/PC13333/PC133330U/Phase-mask-based-lensless-image-reconstruction-using-self-attention/10.1117/12.3042497.short)

The trained models for RML and DiffuserCam reconstructions using ConvNeXt are available from [Google Drive](https://drive.google.com/drive/folders/129L_sWBN9wNy7yLfXQds_5uPFE36QFYO?usp=sharing). Automatic download scripts are provided in the relevant directories within this repository.

## Data Availability

Sec. 5 (Experimental Information Evaluation) uses parallel lensless imaging system measurements captured for images from the MIRFLICKR-25000 dataset. The training data are available at the [Parallel Lensless Dataset website](https://waller-lab.github.io/parallel-lensless-dataset/datasets.html).

Photon count calibration data (Fig. 4c) are also available at the [Parallel Lensless Dataset website](https://waller-lab.github.io/parallel-lensless-dataset/datasets.html).

Mutual information estimates and other data necessary for reproducing figures are available in the corresponding directories within this repository.

## Paper

```bibtex
@article{Kabuli2026LenslessInfo,
author = {Leyla A. Kabuli and Henry Pinkard and Eric Markley and Clara S. Hung and Laura Waller},
journal = {Optica},
keywords = {Computational imaging; Imaging systems; Neural networks; Optical imaging; Systems design; Three dimensional imaging},
number = {2},
pages = {227--235},
publisher = {Optica Publishing Group},
title = {Designing lensless imaging systems to maximize information capture},
volume = {13},
month = {Feb},
year = {2026},
url = {https://opg.optica.org/optica/abstract.cfm?URI=optica-13-2-227},
doi = {10.1364/OPTICA.570334},
}
```

## Contact 

Please reach out to lakabuli@berkeley.edu with any questions or concerns.
