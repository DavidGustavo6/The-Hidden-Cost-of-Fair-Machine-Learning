# Fairness vs Information Retention in Machine Learning

## Overview

This project investigates the trade-off between fairness and information retention in machine learning models. We compare fairness techniques from different methodological families across **5 benchmark datasets**:

- **Pre-processing**: Reweighing, feature masking, disparate impact removal, label flipping, sampling strategies
- **In-processing**: Adversarial debiasing, fairness-aware regularization, constrained optimization
- **Post-processing**: Equalized odds adjustment, threshold optimization, calibration, reject option classification

The goal is to evaluate how these approaches improve equity across demographic groups while potentially reducing predictive performance or discarding valuable information.

## Project Structure

```
IAS/
├── data/                    # Dataset files
├── figures/                 # Generated visualizations
├── results/                 # Experimental results (CSV)
├── notebooks/               # Jupyter notebooks for experiments
│   ├── 01_german_credit_analysis.ipynb
│   ├── 02_compas_analysis.ipynb
│   ├── 03_bank_marketing_analysis.ipynb
│   ├── 04_law_school_analysis.ipynb
│   ├── 05_adult_census_analysis.ipynb
│   └── 06_cross_dataset_comparison.ipynb
├── src/                     # Source code modules
│   ├── datasets.py          # Multi-dataset loader (5 datasets)
│   ├── fairness_metrics.py  # Fairness evaluation metrics
│   ├── preprocessing.py     # Pre-processing fairness methods (5)
│   ├── inprocessing.py      # In-processing fairness methods (3)
│   ├── postprocessing.py    # Post-processing fairness methods (4)
│   └── visualization.py     # Enhanced plotting functions (7)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Datasets

| Dataset | Source | Samples | Sensitive Attr | Task |
|---------|--------|---------|----------------|------|
| **German Credit** | UCI | 1,000 | Sex, Age | Credit risk classification |
| **COMPAS** | ProPublica | ~7,000 | Race, Sex | Recidivism prediction |
| **Bank Marketing** | UCI | ~45,000 | Age, Marital | Subscription prediction |
| **Law School** | LSAC | ~21,000 | Race, Sex | Bar exam passage |
| **Adult Census** | UCI | ~48,000 | Sex, Race | Income prediction (>$50K) |

## Installation

1. Clone this repository or extract the zip file
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Reproducing Results

### Run Individual Dataset Notebooks
Each notebook is independent and can be run separately:

```bash
jupyter notebook notebooks/01_german_credit_analysis.ipynb
```

### Run Cross-Dataset Comparison
After running all 5 dataset notebooks, aggregate results:

```bash
jupyter notebook notebooks/06_cross_dataset_comparison.ipynb
```

Results are saved to `results/` and figures to `figures/`.

## Fairness Methods

### Pre-processing (5 methods)
| Method | Description | Reference |
|--------|-------------|-----------|
| Reweighing | Assigns sample weights for demographic parity | Kamiran & Calders, 2012 |
| Feature Masking | Removes sensitive attributes and proxies | - |
| Disparate Impact Remover | Modifies feature distributions | Feldman et al., 2015 |
| Label Flipping | Selectively flips labels to reduce bias | - |
| Sampling Strategy | Over/under-sampling for balance | - |

### In-processing (3 methods)
| Method | Description | Reference |
|--------|-------------|-----------|
| Fairness Regularization | Adds fairness penalty to loss function | - |
| Adversarial Debiasing | Adversarial network removes bias | Zhang et al., 2018 |
| Constrained Classifier | Projected gradient descent with constraints | - |

### Post-processing (4 methods)
| Method | Description | Reference |
|--------|-------------|-----------|
| Equalized Odds | Adjusts predictions for TPR/FPR parity | Hardt et al., 2016 |
| Threshold Optimizer | Group-specific classification thresholds | - |
| Calibrated Post-Processing | Group-wise probability calibration | - |
| Reject Option Classification | Favors unprivileged in uncertainty region | Kamiran et al., 2012 |

## Metrics

### Fairness Metrics
- **Demographic Parity Difference (DPD)**: |P(Ŷ=1|S=0) - P(Ŷ=1|S=1)|
- **Equalized Odds Difference (EOD)**: Average of TPR and FPR differences
- **Demographic Parity Ratio**: min(P₀/P₁, P₁/P₀)

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **AUC-ROC**: Area under the ROC curve
- **Precision/Recall/F1**: Standard classification metrics

## Visualizations

The `src/visualization.py` module provides:
- Accuracy vs Fairness trade-off scatter plots
- Grouped bar charts by method category
- Radar charts for multi-metric comparison
- Heatmaps for method-dataset performance
- Pareto frontier plots
- Summary dashboards

## References

1. Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination.
2. Feldman, M., et al. (2015). Certifying and removing disparate impact.
3. Hardt, M., et al. (2016). Equality of opportunity in supervised learning.
4. Zhang, B. H., et al. (2018). Mitigating unwanted biases with adversarial learning.
5. Bellamy, R. K., et al. (2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias.

## Author

David Carvalho - Inteligência Artificial e Sociedade

