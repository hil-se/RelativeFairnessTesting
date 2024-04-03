import numpy as np
from scipy.stats import t

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def mae(self):
        return np.sum(np.abs(np.array(self.y) - np.array(self.y_pred)))/len(self.y)


    def RBD(self, s):
        # s is an array of numerical values of a sensitive attribute
        error = np.array(self.y_pred) - np.array(self.y)
        bias = {}
        bias[1] = error[np.where(np.array(s)==1)[0]]
        bias[0] = error[np.where(np.array(s)==0)[0]]
        bias_diff = np.mean(bias[1]) - np.mean(bias[0])
        sigma = np.std(self.y_pred - self.y, ddof=1)
        if sigma:
            bias_diff = bias_diff / sigma
        else:
            bias_diff = 0.0
        return bias_diff


    def RBT(self, s):
        # s is an array of numerical values of a sensitive attribute
        error = np.array(self.y_pred) - np.array(self.y)
        bias = {}
        bias[1] = error[np.where(np.array(s) == 1)[0]]
        bias[0] = error[np.where(np.array(s) == 0)[0]]
        bias_diff = np.mean(bias[1]) - np.mean(bias[0])
        var1 = np.var(bias[1], ddof=1)
        var0 = np.var(bias[0], ddof=1)
        var = var1/len(bias[1])+var0/len(bias[0])
        if var>0:
            bias_diff = bias_diff / np.sqrt(var)
            dof = var ** 2 / ((var1 / len(bias[1])) ** 2 / (len(bias[1]) - 1) + (var0 / len(bias[0])) ** 2 / (
                        len(bias[0]) - 1))
        else:
            bias_diff = 0.0
            dof = 1
        p = t.sf(np.abs(bias_diff), dof)
        return p
