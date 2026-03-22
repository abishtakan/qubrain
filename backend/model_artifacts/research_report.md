# Research-Grade Hybrid Quantum-Classical Training Report

## Task Definition
- Task: GBM mortality-status classification
- Target definition: 1 = Dead, 0 = Alive
- Research framing: Binary mortality-status classification, not time-to-event survival analysis.

## Dataset Summary
- Source: TCGA-GBM (NCI Genomic Data Commons)
- Final labeled cohort: 384 samples
- Class balance: 312 dead / 72 alive
- Holdout split: 307 train / 77 test

## Methodology Upgrades
- Stratified outer holdout split
- Inner 5-fold cross-validation for model selection
- Fold-local feature selection and scaling
- Leak-free class balancing inside training folds only
- Threshold selection on validation data rather than a fixed 0.5 cutoff
- Overfitting analysis using train-vs-holdout metrics

## Selected Hyperparameters
- Top genes: 50
- Qubits: 6
- Layers: 2
- Learning rate: 0.001
- Dropout: 0.1
- Imbalance strategy: class_weight
- Entropy lambda: 0.02
- Final decision threshold: 0.502
- Final epochs: 18

## Cross-Validation Results
- Best inner-CV mean AUC: 0.6925
- Best inner-CV balanced accuracy: 0.6763

Top candidates:
- genes=50, qubits=6, layers=2, imbalance=class_weight, mean_val_auc=0.6925, mean_overfit_gap=0.1251
- genes=50, qubits=4, layers=1, imbalance=smote, mean_val_auc=0.6867, mean_overfit_gap=0.0759
- genes=50, qubits=4, layers=2, imbalance=smote, mean_val_auc=0.6826, mean_overfit_gap=0.1240
- genes=50, qubits=6, layers=2, imbalance=class_weight, mean_val_auc=0.6589, mean_overfit_gap=0.1303
- genes=50, qubits=4, layers=2, imbalance=smote, mean_val_auc=0.6587, mean_overfit_gap=0.1768

## Final Performance
- Train AUC: 0.8463
- Holdout AUC: 0.7494
- Holdout AUC 95% bootstrap CI: [0.5771, 0.9076]
- Holdout PR AUC: 0.8904
- Holdout accuracy: 0.6623
- Holdout balanced accuracy: 0.7103
- Holdout F1: 0.7547
- Holdout precision: 0.9302
- Holdout recall: 0.6349
- Holdout specificity: 0.7857
- Holdout MCC: 0.3267
- Holdout Brier score: 0.1979
- Risk band cutoffs: low <= 0.489, high >= 0.737

## Overfitting Check
- Train AUC minus holdout AUC: 0.0968
- Train balanced accuracy minus holdout balanced accuracy: 0.0531

## Files
- `metadata.json` contains the full machine-readable result summary
- `cv_results.csv` contains the hyperparameter-search table
- `holdout_predictions.csv` contains per-patient holdout predictions
- `test_patients.json` contains randomized UI-ready holdout examples
- `explainability.json` contains global feature-importance rankings

## Explainability
- Local explanations use Integrated Gradients against the median training-cohort baseline.
- Global feature importance is the mean absolute Integrated Gradients attribution on the outer holdout cohort.

Top global features:
- SUPT20HL1: mean_abs_attr=0.0564, mean_signed_attr=0.0267
- age: mean_abs_attr=0.0391, mean_signed_attr=-0.0081
- IL9: mean_abs_attr=0.0329, mean_signed_attr=-0.0265
- RAB6D: mean_abs_attr=0.0247, mean_signed_attr=-0.0129
- ZNF676: mean_abs_attr=0.0215, mean_signed_attr=-0.0200
- NOX4: mean_abs_attr=0.0211, mean_signed_attr=0.0037
- C16orf87: mean_abs_attr=0.0173, mean_signed_attr=0.0036
- RANBP17: mean_abs_attr=0.0170, mean_signed_attr=0.0124
- gender: mean_abs_attr=0.0170, mean_signed_attr=-0.0170
- CYSLTR2: mean_abs_attr=0.0166, mean_signed_attr=-0.0035