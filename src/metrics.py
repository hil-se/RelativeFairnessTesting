import sklearn.metrics
import numpy as np
import scipy.stats
from pdb import set_trace

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def r2(self):
        return sklearn.metrics.r2_score(self.y, self.y_pred)

    def CBD(self, s):
        # s is an array of numerical values of a sensitive attribute
        if len(np.unique(s)) == 2:
            group0 = max(np.unique(s))
            group1 = min(np.unique(s))
            error = np.array(self.y_pred) - np.array(self.y)
            bias = {}
            bias[group0] = error[np.where(s==group0)[0]]
            bias[group1] = error[np.where(s==group1)[0]]
            bias_diff = np.mean(bias[group0]) - np.mean(bias[group1])
        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(self.y)):
                for j in range(len(self.y)):
                    if s[i] - s[j] > 0:
                        diff_pred = self.y_pred[i] - self.y_pred[j]
                        diff_true = self.y[i] - self[j]
                        n += 1
                        bias_diff += diff_pred - diff_true
            bias_diff = bias_diff / n
        sigma = np.std(self.y_pred - self.y)
        if sigma:
            bias_diff = bias_diff / sigma
        else:
            bias_diff = 0.0
        return bias_diff

    def CBT(self, s):
        # s is an array of numerical values of a sensitive attribute
        if len(np.unique(s)) == 2:
            group0 = max(np.unique(s))
            group1 = min(np.unique(s))
            error = np.array(self.y_pred) - np.array(self.y)
            bias = {}
            bias[group0] = error[np.where(s == group0)[0]]
            bias[group1] = error[np.where(s == group1)[0]]
            bias_diff = np.mean(bias[group0]) - np.mean(bias[group1])
            sigma = np.sqrt(np.std(bias[group0])**2/len(bias[group0]) + np.std(bias[group1])**2/len(bias[group1]))
            if sigma:
                bias_diff = bias_diff / sigma
            else:
                bias_diff = 0.0
        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(self.y)):
                for j in range(len(self.y)):
                    if s[i] - s[j] > 0:
                        diff_pred = self.y_pred[i] - self.y_pred[j]
                        diff_true = self.y[i] - self[j]
                        n += 1
                        bias_diff += diff_pred - diff_true
            bias_diff = bias_diff / n
            sigma = np.std(self.y_pred - self.y)
            if sigma:
                bias_diff = bias_diff * np.sqrt(len(s)) / sigma
            else:
                bias_diff = 0.0
        return bias_diff
