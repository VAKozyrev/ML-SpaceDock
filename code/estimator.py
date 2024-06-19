import os
import time
from copy import deepcopy

import numpy as np
from scipy import sparse as ss
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


class CommonEstimator:

    def __init__(self, args):

        self.model = {
        'logreg': LogisticRegression(max_iter=5000, random_state=args.seed),
        'rf': RandomForestClassifier(random_state=args.seed),
        'mlp': MLPClassifier(random_state=args.seed),
        }[args.model]

        self.seed = args.seed

        self.n_folds = args.n_folds


    def fit(self, X, y):

        self.trained_models=list()

        print(f"Training the model {self.model}\n'0':{np.count_nonzero(y==0)}, '1':{np.count_nonzero(y==1)}")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for train_index, test_index in kf.split(y):

            X_train = X[train_index,:]
            y_train = y[train_index]

            self.trained_models.append(deepcopy(self.model.fit(X_train, y_train)))


    def predict(self, X):

        predictions_matrix = np.empty((len(self.trained_models),X.shape[0])) 

        for i, model in enumerate(self.trained_models):

            predictions_matrix[i] = model.predict_proba(X)[:,1]

        y_mean = np.mean(predictions_matrix, axis=0)
        y_var = np.mean((predictions_matrix-y_mean)**2, axis=0)

        return y_mean, y_var