import numpy as np
import random as random

class Perceptron:
    def __init__(self):
        pass
        
    #My Code
    def fit(self,X,y, max_steps = 1000):
        #code here
        #pick a random w
        #x.shape gives number of rows and number of columns
        #x has n observations (rows) and p features (columns)
        p_features, n_samples = X.shape
        w = np.random.rand(p_features) # w are weights
        
        self.history = []

        for step in range(max_steps) :
            # Compute the predicted label
            i = random.randint(0,n_samples-1)
            xi = X[i,:]
            yi = y[i]
            #xi = np.append(X[i], [1])
            wi = w[i,:]
            print(wi.shape)
            w_sq = np.append(wi, [-1])
            # xi = X[i].view().push(1)
            # w_sq = self.w.view().push(-1)
            z = np.dot(xi, w[i])
            if(z.all()*y[i] < 0):
                y_pred = 1
            else:
                y_pred = 0
            print(w.shape)
            print(y[i].shape)
            print(xi.shape)
            w = w + y_pred*y[i]*xi
            score = self.score(X,y)
            if(score == 1):
                break
            self.history.push(score)
    
    def predict(self,X):
        p_features = X.shape[0]
        n_samples = X.shape[1]
        y = []
        for i in range(n_samples):
            xi = np.append(X[i], [1])
            w_sq = np.append(w[i], [-1])
            if(np.dot(xi,w_sq)>=0):
                y.append(y, [1])
                #y.push(1)
            else:
                y.append(y, [0])
        return y
    
    def score(self,X,y):
        p_features = X.shape[0]
        n_samples = X.shape[1]
        y_pred = self.predict(X)
        score = 0
        for i in range(n_samples):
            if(y_pred[i] == y[i]):
                score += 1
        return score/n_samples