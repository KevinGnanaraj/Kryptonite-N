
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    X_train, y_train, X_test, y_test = load_data(n=9, val=False)

    # Preprocess Data (Features only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # Standardizing Features
    X_test = scaler.transform(X_test)

    pca = PCA(n_components = 0.95)
    # pca = PCA(n_components = 0.6)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test) 

    tree = RandomForestClassifier(n_estimators=200, random_state=50, n_jobs=-1,  max_depth = None, min_samples_leaf= 1, min_samples_split=2)
   
    tree.fit(X_train, y_train)       ## Make predictions on the test set
    y_pred = tree.predict(X_test)


    # Evaluate the model
    print(f"Validation accuracy: {accuracy_score(y_test, y_pred)}")




train_model()
    
