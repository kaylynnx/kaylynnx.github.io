import numpy as np
import random as random

class Perceptron:

    def __init__(self):
         pass
    
    def fit(self, X, y, max_steps = 1000):
        n_samples, p_features = X.shape
        X_sq = np.column_stack([np.ones(n_samples), X])
        self.w = np.zeros(p_features+1)
        self.history = []
        
        for step in range(max_steps):
            pred = np.sign(X_sq.dot(self.w))
            error = (y != pred).astype(int)
            accuracy = 1 - np.mean(error)
            self.history.append(accuracy)
            
            if np.all(pred == y):
                break
                
            index = np.random.choice(range(n_samples), size = 1)
            self.w += error[index]*X_sq[index,:].reshape(-1)

    def predict(self, X):
        n_samples, p_features = X.shape
        X_sq = np.column_stack([np.ones(n_samples), X])
        return ((X_sq.dot(self.w)) >= 0).astype(int)

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
