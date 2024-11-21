
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



def load_data(n, val=False):
    X = np.load('Datasets/kryptonite-%s-X.npy'%(n))
    y = np.load('Datasets/kryptonite-%s-y.npy'%(n))

    if val: # If using validation
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)

        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

        return X_train, y_train, X_test, y_test, X_val, y_val
    
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        return X_train, y_train, X_test, y_test
    
    


def train_model():

    # Load Data & Split the dataset into training and testing sets
    n=15
    X_train, y_train, X_test, y_test = load_data(n, val=False)

    # Feature Selection (Keep 100 best features based on statistical tests)

    i=11
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test) 

    rf_tree = RandomForestClassifier(n_estimators=2000, random_state=50, n_jobs=-1)


    rf_tree.fit(X_train_pca, y_train)       ## Make predictions on the test set
    y_pred = rf_tree.predict(X_test_pca)

    # Evaluate the model
    print(f"Validation accuracy: {accuracy_score(y_test, y_pred)} for n={n} and n_components = {i}")


    # Accuracy for different levels of dimensionality

    # n_values = [9, 12, 15, 18]
    # accuracy = [0.959, 0.945, 0.806, 0.5894]






train_model()

    
