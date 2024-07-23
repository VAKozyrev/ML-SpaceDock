import os
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from models import BinaryClassifierNN

class CommonEstimator:

    def __init__(
        self,
        size: int,
        model: str,
        input_dim: int,
        n_folds: int,
        seed
    ):
        self.size = size
        self.seed = seed
        self.n_folds = n_folds
        self.model_name = model
        self.input_dim = input_dim
        self.model = {
                    'logreg': LogisticRegression(max_iter=5000, random_state=self.seed),
                    'rf': RandomForestClassifier(random_state=self.seed),
                    'mlp': MLPClassifier(random_state=self.seed),
                    'mlp_pytorch': BinaryClassifierNN(input_dim, random_state=self.seed)
        }[model]
        self.predictions_matrix = np.empty((self.n_folds,size)) 

        print(self)

    def __str__(self):
        return f'\tEstimator(\n\
            size: {self.size}\n\
            number of folds: {self.n_folds}\n\
            model: {self.model_name}\n\
            input dimentionality: {self.input_dim}\n\
            random state: {self.seed}\n\t)'

    def fit(self, X, y):

        self.trained_models=list()

        print(f"Training the model {self.model}\n")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for train_index, test_index in kf.split(y):

            X_train = X[train_index,:]
            y_train = y[train_index]

            self.trained_models.append(deepcopy(self.model.fit(X_train, y_train)))


    def predict(self, X):

        for i, model in enumerate(self.trained_models):
            if self.model_name == 'mlp_pytorch':
                self.predictions_matrix[i] = model.predict_proba(X)
            else:
                self.predictions_matrix[i] = model.predict_proba(X)[:,1]

        return self.predictions_matrix.mean(axis=0), self.predictions_matrix.std(axis=0)