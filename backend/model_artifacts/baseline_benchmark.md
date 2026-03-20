# Classical Baseline Benchmark

The baselines below were evaluated on the training partition only using the same fold-local feature-selection and scaling policy as the hybrid model.

- Random Forest: AUC=0.6549, PR AUC=0.8936, Balanced Acc=0.4922, F1=0.8810
- SVM (RBF): AUC=0.6059, PR AUC=0.8789, Balanced Acc=0.5091, F1=0.8681
- Logistic Regression: AUC=0.6033, PR AUC=0.8640, Balanced Acc=0.5522, F1=0.7653
- Classical MLP: AUC=0.5735, PR AUC=0.8454, Balanced Acc=0.5740, F1=0.8287