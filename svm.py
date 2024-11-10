import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def load_data(n, val=False):
    X = np.load('Datasets/kryptonite-%s-X.npy'%(n))
    y = np.load('Datasets/kryptonite-%s-y.npy'%(n))

    if val:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)

        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

        # print(f"Checking shapes of all the data: {X.shape}, {X_train.shape}, {X_test.shape}, {X_val.shape}")
        # print(f"Checking y shapes: {y.shape}, {y_train.shape}, {y_test.shape}, {y_val.shape}")

        return X_train, y_train, X_test, y_test, X_val, y_val
    
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        return X_train, y_train, X_test, y_test

def main():
    n_values = [9, 12, 15, 18, 24, 30, 45]

    # X_train, y_train, X_test, y_test, X_val, y_val = load_data(9)
    X_train, y_train, X_test, y_test = load_data(9, val=False)

    pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('svm', SVC(max_iter=10000))
        ]
    )

    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['rbf', 'poly'],
        'svm__gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    y_test_pred = grid_search.predict(X_test)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Validation accuracy: {accuracy_score(y_test, y_test_pred)}")

main()

# Best params found: {'svm__C': 10, 'svm__gamma': 'scale', 'svm__kernel': 'rbf'}