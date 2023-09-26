import numpy as np
from scipy.stats import t

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    # def mse(self):
    #     return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return np.sum(np.abs(np.array(self.y) - np.array(self.y_pred)))/len(self.y)
        # return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    # def r2(self):
    #     return sklearn.metrics.r2_score(self.y, self.y_pred)

    def RBD(self, s):
        # s is an array of numerical values of a sensitive attribute
        if len(np.unique(s)) == 2:
            error = np.array(self.y_pred) - np.array(self.y)
            bias = {}
            bias[1] = error[np.where(np.array(s)==1)[0]]
            bias[0] = error[np.where(np.array(s)==0)[0]]
            bias_diff = np.mean(bias[1]) - np.mean(bias[0])

            # var1 = np.var(bias[1], ddof=1)
            # var0 = np.var(bias[0], ddof=1)
            # var = (var1 * (len(bias[1])-1) + var0 * (len(bias[0])-1))/(len(bias[1])+len(bias[0])-2)

        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(self.y)):
                for j in range(len(self.y)):
                    if np.array(s)[i] - np.array(s)[j] > 0:
                        diff_pred = self.y_pred[i] - self.y_pred[j]
                        diff_true = self.y[i] - self.y[j]
                        n += 1
                        bias_diff += diff_pred - diff_true
            bias_diff = bias_diff / n
        sigma = np.std(self.y_pred - self.y, ddof=1)
        if sigma:
            bias_diff = bias_diff / sigma
        else:
            bias_diff = 0.0
        return bias_diff


    def RBT(self, s):
        # s is an array of numerical values of a sensitive attribute
        if len(np.unique(s)) == 2:
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
        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(self.y)):
                for j in range(len(self.y)):
                    if np.array(s)[i] - np.array(s)[j] > 0:
                        diff_pred = self.y_pred[i] - self.y_pred[j]
                        diff_true = self.y[i] - self.y[j]
                        n += 1
                        bias_diff += diff_pred - diff_true
            bias_diff = bias_diff / n
            sigma = np.std(self.y_pred - self.y, ddof = 1)
            if sigma:
                bias_diff = bias_diff * np.sqrt(len(s)) / sigma
            else:
                bias_diff = 0.0
            dof = len(s)-1
        p = t.sf(np.abs(bias_diff), dof)
        return p
