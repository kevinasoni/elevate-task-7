# elevate-task-7

Title: SVM Classification on Breast Cancer Dataset

Objective
The objective of this project is to perform binary classification using Support Vector Machines (SVMs) with linear and non-linear kernels. The project includes data preprocessing, training SVM models, visualizing decision boundaries, tuning hyperparameters, and evaluating performance using cross-validation.

Dataset
The dataset used is:
breast-cancer.csv
(Loaded locally from the provided file path.)

The dataset contains diagnostic data where the target variable is:

• M → Malignant (1)
• B → Benign (0)

Project Workflow

Load and Prepare Dataset
• Import dataset using pandas.
• Drop irrelevant columns (id).
• Convert target labels (diagnosis) from M/B to 1/0.
• Select two features (radius_mean, texture_mean) for 2D visualization.

Feature Scaling
StandardScaler is used to normalize the selected features.

Train SVM Models
Two SVM classifiers are trained:

• Linear SVM (kernel = 'linear')
• Non-linear SVM using RBF kernel (kernel = 'rbf')

Decision Boundary Visualization
Using 2D grids and contour plots, decision boundaries for both SVM models are visualized.

Hyperparameter Tuning
GridSearchCV is applied to RBF SVM with:

• C values: [0.1, 1, 10, 50]
• gamma values: [0.001, 0.01, 0.1, 1]

The best parameters and cross-validation score are obtained.

Cross-Validation
Cross-validation accuracy is computed using GridSearchCV’s scoring.

Results
• Best parameters for RBF SVM:
C = 0.1
gamma = 1

• Best cross-validation accuracy ≈ 89.63%
