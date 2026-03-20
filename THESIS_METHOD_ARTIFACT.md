# Thesis Method Artifact

This artifact is the current thesis-ready summary of the `qubrain` research pipeline. It uses the latest corrected saved artifacts in `qubrain/backend/model_artifacts/` and supersedes older notes where numbers or implementation details drifted.

## 1. Dataset (Source And Size)

- **Dataset source:** The Cancer Genome Atlas Glioblastoma Multiforme cohort (`TCGA-GBM`) from the NCI Genomic Data Commons (`GDC`).
- **Modalities used:** RNA-seq gene-expression data plus clinical metadata.
- **Raw downloaded expression files discovered:** `391` STAR gene-count files.
- **Usable modeled expression matrix after loading, filtering, and alignment:** `384` samples x `19,962` protein-coding genes.
- **Final labeled cohort used for training/testing:** `384` patients.
- **Class distribution:** `312 Dead` and `72 Alive`.
- **Class ratio:** `81.25%` Dead and `18.75%` Alive.

Important framing note:

- This project is a **mortality-status classification** study using `demographic.vital_status`.
- It is **not** a true time-to-event survival-analysis model with censoring.

## 2. Class Balancing (Classification Problems)

This is an imbalanced binary-classification problem, so class handling is required.

- **Explored balancing strategies:** `SMOTE` and `class_weight`.
- **Scientifically correct leakage policy:** balancing is applied only inside the training fold or the final training split, never before the outer split and never on the holdout set.
- **Final selected strategy:** `class_weight`.
- **Loss used in the final model:** entropy-regularized binary cross-entropy with per-class weighting.

Why this matters:

- If imbalance correction is applied before splitting, the evaluation becomes optimistic.
- Because the dataset is imbalanced, `AUC`, `balanced accuracy`, `F1`, `recall`, and `specificity` are more informative than raw accuracy alone.

## 3. Tokenization, Vectorization, And Vectorization Techniques

This section is **not applicable as NLP tokenization**, because the project is not a text-processing task.

The correct equivalent for this project is **feature engineering and numerical representation**:

- RNA-seq values are loaded from tabular gene-count files.
- Only `protein_coding` genes are retained.
- Expression is transformed as `log2(TPM + 1)`.
- `age` is treated as a numeric feature.
- `gender` is binary encoded.
- Univariate feature selection is applied with `SelectKBest(f_classif)`.
- The final model vector is:
  - `[age, gender, selected_gene_1, ..., selected_gene_50]`
- Feature scaling is performed with `MinMaxScaler` fitted on training data only.

## 4. Layers And Activation Functions (NN Project)

The final deployed model is a **hybrid quantum-classical neural network**.

### Final selected architecture

- **Input dimension:** `52`
  - `2` clinical features
  - `50` selected genes
- **Classical encoder:**
  - `Linear(52, 64)`
  - `ReLU`
  - `Dropout(0.1)`
  - `Linear(64, 6)`
  - `Tanh`
- **Quantum layer:**
  - `6` qubits
  - `2` variational layers
  - `AngleEmbedding`
  - `StronglyEntanglingLayers`
- **Prediction head:**
  - `Linear(6, 32)`
  - `ReLU`
  - `Linear(32, 1)`
  - `Sigmoid(logits / 0.35)`

Why these activations are appropriate:

- `ReLU` provides stable nonlinear learning in the classical encoder and head.
- `Tanh` bounds the encoder outputs before conversion to quantum rotation angles.
- `Sigmoid` is appropriate for binary probability output.

## 5. Selection Of Models (Including Pre-Trained Models)

### Pre-trained models

- **No pre-trained models were used.**
- This is a small tabular biomedical prediction problem, so all models were trained from scratch on the project dataset.

### Models compared

The current package includes classical baseline benchmarking using the same leak-free training-partition protocol:

- `Random Forest`
- `SVM (RBF)`
- `Logistic Regression`
- `Classical MLP`
- `Hybrid quantum-classical classifier`

### Why the hybrid model was selected

- It is the central research contribution of the project.
- Under the saved current benchmark protocol, it achieved the strongest mean validation AUC among the evaluated pipelines.

Current mean CV AUC comparison:

- **Hybrid quantum-classical:** `0.6925`
- `Random Forest`: `0.6549`
- `SVM (RBF)`: `0.6059`
- `Logistic Regression`: `0.6033`
- `Classical MLP`: `0.5735`

## 6. Final Model Used

- **Final model:** Hybrid quantum-classical binary classifier
- **Task:** GBM mortality-status classification
- **Target definition:** `1 = Dead`, `0 = Alive`
- **Selected genes:** `50`
- **Selected hyperparameters:**
  - `n_top_genes = 50`
  - `n_qubits = 6`
  - `n_layers = 2`
  - `learning_rate = 0.001`
  - `dropout = 0.1`
  - `batch_size = 32`
  - `max_epochs = 60`
  - `patience = 10`
  - `entropy_lambda = 0.02`
  - `imbalance_strategy = class_weight`
  - `hidden_dim = 64`
  - `head_dim = 32`
  - `temperature = 0.35`
  - `weight_decay = 0.0`
  - `decision_threshold = 0.502`

## 7. Train-Test Split

- **Outer split strategy:** stratified `80/20` train-test split
- **Training set:** `307` samples
- **Holdout test set:** `77` samples
- **Training distribution:** `249 Dead`, `58 Alive`
- **Holdout distribution:** `63 Dead`, `14 Alive`

Why stratification was used:

- To keep the class ratio approximately stable across training and holdout partitions.

## 8. Hyperparameter Tuning

Hyperparameter tuning was performed on training data only.

- **Tuning protocol:** inner stratified `5-fold` cross-validation on the training partition
- **Selection metric:** mean validation `ROC-AUC`
- **Threshold selection:** best validation `balanced accuracy` threshold per fold

### Search dimensions explored

- number of selected genes
- number of qubits
- number of quantum layers
- learning rate
- dropout
- entropy regularization strength
- imbalance strategy
- hidden dimension
- head dimension
- temperature scaling
- weight decay
- patience and maximum epochs

### Top ranked current configuration

- `50` genes, `6` qubits, `2` layers, `class_weight`
- mean train AUC: `0.8176`
- mean validation AUC: `0.6925`
- mean validation balanced accuracy: `0.6763`
- mean overfitting gap AUC: `0.1251`
- mean selected epoch: `18.4`
- mean selected threshold: `0.502`

## 9. Model Testing And Evaluation

The final model was retrained on the full training split using the selected hyperparameters and then evaluated once on the untouched holdout set.

### Metrics reported

- `ROC-AUC`
- `PR-AUC`
- `Accuracy`
- `Balanced Accuracy`
- `F1-score`
- `Precision`
- `Recall`
- `Specificity`
- `Matthews Correlation Coefficient (MCC)`
- `Brier score`
- `95%` bootstrap confidence interval for holdout AUC

Why `AUC` is the headline metric:

- It is threshold-independent.
- It is more reliable than raw accuracy on imbalanced binary data.

## 10. Evaluation Scores

### Cross-validation summary

- best inner-CV mean AUC: `0.6925`
- best inner-CV balanced accuracy: `0.6763`

### Final training-set metrics

- train AUC: `0.8462`
- train PR-AUC: `0.9532`
- train accuracy: `0.7557`
- train balanced accuracy: `0.7634`
- train F1: `0.8330`
- train precision: `0.9350`
- train recall: `0.7510`
- train specificity: `0.7759`
- train MCC: `0.4328`
- train Brier score: `0.1648`

### Final holdout-set metrics

- holdout AUC: `0.7494`
- holdout PR-AUC: `0.8904`
- holdout accuracy: `0.6623`
- holdout balanced accuracy: `0.7103`
- holdout F1: `0.7547`
- holdout precision: `0.9302`
- holdout recall: `0.6349`
- holdout specificity: `0.7857`
- holdout MCC: `0.3267`
- holdout Brier score: `0.1978`
- holdout AUC 95% bootstrap CI: `[0.5771, 0.9076]`

### Holdout confusion matrix

- `TN = 11`
- `FP = 3`
- `FN = 23`
- `TP = 40`

## 11. Cross Validation

Cross-validation was used exactly as follows:

- outer holdout split for final testing
- inner `5-fold StratifiedKFold` on the training partition for model selection
- feature selection fitted within each fold only
- scaling fitted within each fold only
- class balancing applied within each fold only

This is the correct approach for small, high-dimensional biomedical datasets because it reduces information leakage and gives a more honest estimate of generalization during model selection.

## 12. Model Overfitting And Model Underfitting

### Overfitting

Current overfitting evidence:

- train AUC: `0.8462`
- holdout AUC: `0.7494`
- AUC gap: `0.0968`
- train balanced accuracy: `0.7634`
- holdout balanced accuracy: `0.7103`
- balanced-accuracy gap: `0.0531`

Interpretation:

- The model shows a **moderate train-to-holdout gap**, but not a catastrophic one.
- In the saved metadata this is classified as `acceptable_generalization`.
- This means some overfitting exists, but the model still transfers a meaningful amount of signal to unseen patients.

Why overfitting risk is naturally high in this project:

- there are many genes relative to the number of patients
- the dataset is imbalanced
- hybrid models can memorize small datasets if preprocessing and validation are not handled carefully

### Underfitting

Current underfitting interpretation:

- The model is **not severely underfitting**, because the training AUC is high (`0.8462`).
- However, the inner-CV AUC (`0.6925`) and holdout AUC (`0.7494`) show that the predictive signal is still limited and the task remains difficult.
- This means the model still has some bias and uncertainty, even after tuning.

What was already done to reduce underfitting:

- broadened the search space beyond a minimal MVP model
- increased hidden dimensions from the smaller default architecture
- tested deeper and wider quantum/classical configurations
- tuned dropout, entropy regularization, and temperature
- compared `SMOTE` and `class_weight`
- tuned the decision threshold on validation data instead of fixing it at `0.5`

What remains for a stronger thesis beyond the current package:

- repeated or nested outer evaluation rather than one holdout split
- ablation study on clinical-only, genes-only, and hybrid variants
- external validation on a second cohort if available

## 13. Files That Support This Artifact

- `qubrain/backend/model_artifacts/metadata.json`
- `qubrain/backend/model_artifacts/cv_results.csv`
- `qubrain/backend/model_artifacts/holdout_predictions.csv`
- `qubrain/backend/model_artifacts/research_report.md`
- `qubrain/backend/model_artifacts/baseline_benchmark.csv`
- `qubrain/backend/model_artifacts/explainability.json`

## 14. One-Line Thesis Summary

This project uses a leak-aware, cross-validated **hybrid quantum-classical model** to classify **GBM mortality status** from clinical and RNA-seq features, achieving a current holdout `ROC-AUC` of `0.7494` on a stratified unseen test set.

## 15. Explainability

Explainability is now part of the saved `qubrain` package.

- **Local explainability method:** Integrated Gradients for each assessed patient
- **Reference baseline:** median feature profile from the training cohort
- **Global explainability method:** mean absolute Integrated Gradients attribution over the outer holdout cohort
- **Deployment artifact:** `explainability.json`

What this means in the UI:

- the clinician can still see the risk-band result first
- the system now also shows the top factors that increased the model's risk score
- and the top factors that reduced the model's risk score

Important scientific caution:

- these are **model attributions**, not causal biomedical claims
- they explain what drove the trained model's prediction relative to the cohort reference profile
