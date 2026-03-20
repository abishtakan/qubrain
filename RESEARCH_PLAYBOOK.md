# Research Playbook For The Hybrid Quantum-Classical GBM Project

This document is the thesis/viva cheat sheet for this project. It is written to be scientifically honest, specific to the codebase, and safe to defend.

## 1. Project Framing

The most important scientific rule is to frame the task correctly.

- Current task in this repository: **GBM mortality-status classification**
- Target variable: `1 = Dead`, `0 = Alive`
- Data source for the label: `demographic.vital_status`
- This is **not** true survival analysis, because the deployed target is not time-to-event with censoring.

If you call this "survival prediction" in the thesis, an examiner can challenge you. The correct wording is:

> "This project predicts mortality status in the TCGA-GBM cohort using clinical and gene-expression features. It is a binary classification study, not a time-to-event survival model."

If you later want a real survival-analysis thesis, you must switch the target to fields such as `days_to_death` / follow-up time and use survival-specific methods.

## 2. Dataset (Source And Size)

Use this wording:

- **Dataset source:** The Cancer Genome Atlas Glioblastoma Multiforme cohort (TCGA-GBM), downloaded from the NCI Genomic Data Commons (GDC).
- **Modality used:** RNA-seq gene expression plus clinical metadata.
- **Raw expression matrix after loading:** 391 RNA-seq files x 19,962 protein-coding genes.
- **Final aligned labeled cohort:** 384 patients/samples.
- **Class distribution:** 312 `Dead`, 72 `Alive`.
- **Class ratio:** 81.25% dead, 18.75% alive.

Why the number drops from 391 to 384:

- some files do not align cleanly with the case mapping and usable clinical labels
- only samples with both expression data and a valid binary vital-status label are kept

## 3. Class Balancing (Classification Problems)

This project is strongly imbalanced, so class handling is not optional.

Scientifically correct options:

- `SMOTE` applied **only inside the training fold**
- class-weighted loss
- threshold tuning on validation data
- reporting imbalance-aware metrics such as AUC, F1, precision, recall, and specificity

What is correct to say:

> "Because the positive class dominated the cohort, class imbalance was handled only within the training data to avoid information leakage."

What you must never do:

- oversample before train/test split
- oversample before cross-validation splitting
- tune thresholds on the test set

Important note for this repository:

- the current `qubrain` trainer does **not** yet use SMOTE or a weighted loss
- this is one reason the hybrid AUC is lower than the older research scripts

## 4. Tokenization / Vectorization / Vectorization Techniques

This section is **not directly applicable** because your project is not NLP.

If an examiner asks this, the correct answer is:

> "This is not a text-processing problem, so tokenization is not used. The equivalent step is feature construction from structured clinical variables and gene-expression measurements."

What the project uses instead:

- expression values are loaded from RNA-seq count files
- only protein-coding genes are kept
- expression is transformed with `log2(TPM + 1)`
- age is numeric
- gender is binary encoded
- the final feature vector is:
  - `[age, gender, selected_gene_1, ..., selected_gene_k]`

So your "vectorization" section in the thesis should really be titled:

- **Feature Engineering And Numerical Representation**

## 5. Feature Selection

Feature selection is central in this project because there are 19,962 genes but only 384 usable samples.

Current production approach:

- remove zero-variance genes
- apply `SelectKBest(f_classif)` on the training data
- keep the top `50` genes

Scientifically correct rule:

> Feature selection must be fitted only on the training fold and then applied to the validation/test fold.

That rule matters because gene-expression data is high-dimensional and extremely vulnerable to leakage.

## 6. Layers And Activation Functions (Neural Network Project)

Your deployed hybrid model is a classical encoder + variational quantum circuit + classical prediction head.

Current architecture:

- **Input layer:** 52 features total
  - 2 clinical features
  - 50 selected genes
- **Classical encoder:**
  - `Linear(n_features, 32)`
  - `ReLU`
  - `Dropout(0.2)`
  - `Linear(32, n_qubits)`
  - `Tanh`
- **Quantum layer:**
  - `AngleEmbedding`
  - `StronglyEntanglingLayers`
  - `6` qubits
  - `2` circuit layers
- **Prediction head:**
  - `Linear(6, 16)`
  - `ReLU`
  - `Linear(16, 1)`
  - `Sigmoid` with temperature scaling

Why these activations make sense:

- `ReLU` is simple and stable for the classical encoder/head
- `Tanh` compresses encoder outputs into a bounded range before converting to quantum rotation angles
- `Sigmoid` is appropriate for binary classification probabilities

## 7. Selection Of Models

This is where you justify why the hybrid model belongs in the thesis even if a classical baseline is competitive.

The correct research logic is:

1. define the research question
2. compare the hybrid model against strong classical baselines
3. keep the hybrid model as the main thesis contribution
4. report honestly whether it beats the baselines

The older benchmark artifacts in this repository show the following mean 5-fold AUC values:

- Classical MLP: `0.7369`
- Logistic Regression: `0.7294`
- SVM (RBF): `0.6776`
- Random Forest: `0.6456`
- XGBoost: `0.6398`

So the defensible thesis statement is:

> "The hybrid quantum-classical model was selected as the primary research model because it is the novel contribution of the study, while classical baselines were used to establish whether the quantum approach offers competitive predictive performance."

### If We Use Pre-Trained Models

This project does **not** use pre-trained foundation models.

So the correct answer is:

> "No pre-trained models were used. All models were trained on the project dataset because the task is a small tabular biomedical prediction problem rather than a transfer-learning scenario in vision or NLP."

## 8. Final Model Used

For the thesis, the final model should be described as:

- **Final research model:** Hybrid quantum-classical binary classifier
- **Input:** age, gender, and top-ranked gene-expression features
- **Output:** probability of mortality status (`Dead` vs `Alive`)

Important nuance:

- the final deployed artifact in `qubrain` is currently the hybrid model
- however, the current production trainer is a simplified version of the research idea, not yet the strongest possible research pipeline

## 9. Train-Test Split

Current production split:

- stratified `80/20` split
- training set: `307` samples
- holdout test set: `77` samples

Training-set class distribution:

- `249` dead
- `58` alive

Holdout-set class distribution:

- `63` dead
- `14` alive

What is scientifically correct to say:

> "A stratified train-test split was used so that the class ratio remained approximately constant in the training and test partitions."

## 10. Cross Validation

Cross-validation is required because the dataset is small.

Current production setup:

- `5-fold StratifiedKFold` on the training portion
- fold-local feature selection
- fold-local scaling

Best-practice thesis setup:

- keep an untouched outer holdout set for final reporting
- use inner stratified 5-fold cross-validation on the training set for model comparison and tuning

If you want the most rigorous option, use:

- **nested cross-validation** for hyperparameter tuning and model selection

That is stronger than a simple single holdout because it reduces the chance of reporting a lucky split.

## 11. Hyperparameter Tuning

Your older research scripts already explored some of the right tuning dimensions:

- number of selected genes: `50`, `100`
- number of qubits: `4`, `6`, `8`
- number of variational layers: `1`, `2`
- learning rate: `0.001`, `0.0005`

One recorded quick-search result in this repository was:

- `50 genes`, `4 qubits`, `2 layers`, `lr=0.001` -> AUC `0.7041`

What a thesis-grade tuning section should say:

> "Hyperparameters were tuned on training data only, using cross-validation. The search space included feature count, quantum circuit width and depth, and optimizer learning rate."

What is not acceptable:

- using the test fold for early stopping
- using the test fold as a tuning proxy
- reporting the best lucky run without describing the search protocol

## 12. Model Testing And Evaluation

Because the classes are imbalanced, accuracy alone is not enough.

Primary metric:

- **ROC-AUC**

Secondary metrics:

- accuracy
- F1-score
- precision
- recall / sensitivity
- specificity
- Brier score

Why AUC should be the headline:

- it is threshold-independent
- it is more informative than raw accuracy in imbalanced binary classification

Why accuracy should not be the headline:

- a model can look "accurate" in an imbalanced dataset while still discriminating poorly

## 13. Evaluation Scores

### Current Production Hybrid Scores

From the current `qubrain` artifacts:

- train-CV AUC: `0.6292`
- holdout AUC: `0.6531`
- holdout accuracy: `0.8182`
- holdout F1: `0.9000`
- holdout precision: `0.8182`
- holdout recall: `1.0000`
- holdout Brier score: `0.1553`

### Why The AUC Dropped

The current production trainer is stricter than the older scripts:

- no SMOTE
- no class-weighted loss
- simpler training recipe
- one specific holdout split
- no strong nested tuning yet

So the lower AUC is not automatically a model failure. It is partly a sign that the current result is more conservative.

### Older Results In This Repository

There are older higher numbers in the repository, but they must be handled carefully:

- older hybrid experiments reported around `0.75` mean AUC
- corrected scientific baseline `v9` reported mean AUC `0.7297`

Do **not** simply quote the highest number in the repo. Quote the number that comes from the most scientifically defensible protocol.

## 14. Overfitting And Underfitting

You must be able to explain both in words and in relation to your own results.

### Overfitting

Definition:

- the model learns patterns that are specific to the training data but do not generalize

How to detect it:

- training AUC is much higher than validation/test AUC
- validation performance drops while training performance keeps improving
- fold-to-fold performance is unstable

In this project, overfitting risk is high because:

- there are many features relative to the number of patients
- quantum models can still memorize small datasets if not regularized properly
- tuning on the wrong split can make results look better than they really are

### Underfitting

Definition:

- the model is too simple or too constrained to learn the signal

How to detect it:

- both training and validation performance are low
- loss stops improving early
- model performance barely exceeds baseline models

In this project, underfitting can happen if:

- too few genes are selected
- the quantum circuit is too shallow
- regularization is too strong
- the optimizer learning rate is too low or the number of epochs is too small

## 15. What A Strong Thesis Must Include

If you want a research project that feels complete and defensible, the final methodology should include all of the following:

- correct task framing: mortality classification, not survival analysis
- exact dataset source and final usable sample count
- explicit class imbalance analysis
- fold-local preprocessing and feature selection
- class balancing only inside training folds
- comparison against strong classical baselines
- hyperparameter tuning on training data only
- one untouched final holdout set or nested cross-validation
- multiple evaluation metrics, with AUC as the primary metric
- overfitting analysis
- ablation study
- explainability analysis
- limitations and threats to validity

## 16. Ablation Study You Should Add

To make the thesis stronger, add an ablation table with:

- clinical only
- genes only
- clinical + genes
- hybrid without SMOTE
- hybrid with SMOTE
- hybrid without entropy regularization
- hybrid with entropy regularization
- `50` genes vs `100` genes
- `4` qubits vs `6` qubits

This lets you answer:

- which part of the pipeline actually helps?
- is the quantum part adding value?
- is performance coming from preprocessing rather than architecture?

## 17. What To Say In The Viva

If asked "What dataset did you use?":

> "I used the TCGA-GBM cohort from the NCI Genomic Data Commons. After aligning gene-expression files with clinical labels and filtering to valid Alive/Dead status, the final cohort contained 384 samples with 19,962 protein-coding genes before feature selection."

If asked "How did you handle imbalance?":

> "The data was strongly imbalanced, with about 81% dead and 19% alive cases. Therefore, class-balancing methods such as SMOTE should only be applied inside training folds to avoid leakage, and evaluation should emphasize AUC, F1, precision, recall, and specificity rather than accuracy alone."

If asked "Why a hybrid quantum model?":

> "The hybrid quantum-classical model is the novel contribution of the work. Classical baselines were included to verify whether the hybrid approach is competitive on small, high-dimensional biomedical data."

If asked "Why not tokenization?":

> "This is not an NLP task. Instead of tokenization, the equivalent step is numerical feature construction from clinical variables and transformed gene-expression values."

If asked "How did you avoid leakage?":

> "Feature selection, scaling, and any class balancing must be fitted on training data only within each fold, then applied to validation or test data."

## 18. The Biggest Scientific Mistakes To Avoid

Do not make any of these claims unless you truly support them:

- "survival prediction" when the label is only Alive/Dead
- "state-of-the-art" without a fair benchmark
- "quantum advantage" without clearly beating classical models
- "best model" if hyperparameters were tuned on the test data
- "generalizable" without external validation or at least strong repeated/nested validation

## 19. The Most Honest Current Status Of This Repo

Right now, the repository contains:

- a scientifically cleaner but simpler production hybrid pipeline
- older scripts with stronger reported numbers but weaker evaluation discipline

So the honest position is:

> "The project already demonstrates a complete hybrid quantum-classical pipeline for GBM mortality classification, but the final thesis version should strengthen tuning, imbalance handling, and validation protocol before making strong performance claims."

## 20. What To Do Next

If the goal is a thesis-grade final version, the next engineering tasks should be:

1. upgrade the `qubrain` trainer to include fold-local SMOTE or weighted loss
2. restore entropy regularization in a clean and leak-free way
3. run nested cross-validation for tuning
4. compare `50` vs `100` genes and `4` vs `6` qubits
5. generate an ablation table and confidence intervals
6. rerun the final hybrid model and save a fresh thesis-ready report

That is the path to a project that is not just deployable, but academically defensible.
