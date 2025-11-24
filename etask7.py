import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load dataset and select two features
data = load_breast_cancer()
X = data.data[:, :2]  # select first two features for 2D demo
y = data.target       # binary: 0=malignant, 1=benign

# 2. Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3a. Linear SVM
linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X_train_scaled, y_train)
print("Linear SVM test accuracy:", linear_svm.score(X_test_scaled, y_test))

# 3b. RBF SVM (with hyperparameter tuning)
params = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 'scale']}
grid = GridSearchCV(SVC(kernel='rbf'), params, cv=5)
grid.fit(X_train_scaled, y_train)
print("RBF SVM best params:", grid.best_params_)
print("RBF SVM test accuracy:", grid.score(X_test_scaled, y_test))

# 4. Cross-validation scores
cv_scores_linear = cross_val_score(SVC(kernel='linear', C=1), scaler.transform(X), y, cv=5)
print("Linear CV mean accuracy:", cv_scores_linear.mean())
cv_scores_rbf = cross_val_score(SVC(kernel='rbf', **grid.best_params_), scaler.transform(X), y, cv=5)
print("RBF CV mean accuracy:", cv_scores_rbf.mean())

# 5. Plot decision boundary (utility)
def plot_boundary(clf, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title(title)
    plt.show()

plot_boundary(linear_svm, X_train_scaled, y_train, "Linear SVM Train Boundary")
plot_boundary(grid.best_estimator_, X_train_scaled, y_train, "RBF SVM Train Boundary")
