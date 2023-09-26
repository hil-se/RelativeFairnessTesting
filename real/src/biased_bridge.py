import numpy as np
from scipy.stats import t

class BiasedBridge:
    def __init__(self, delta_train, delta_test):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.delta_train = delta_train
        self.delta_test = delta_test

    def norm_stats(self, x):
        mu = np.mean(x)
        var = np.var(x, ddof=1)
        return mu, var

    def stats(self, group_train, group_test):
        mu_train, var_train = self.norm_stats(np.array(self.delta_train)[group_train])
        mu_test, var_test = self.norm_stats(np.array(self.delta_test)[group_test])
        mu = mu_test - mu_train
        var = var_train + var_test
        return mu, var

    def RBT(self, s_train, s_test):
        if len(np.unique(s_train)) == 2 and len(np.unique(s_test)) == 2:
            group0_train = np.where(np.array(s_train) == 0)[0]
            group0_test = np.where(np.array(s_test) == 0)[0]
            group1_train = np.where(np.array(s_train) == 1)[0]
            group1_test = np.where(np.array(s_test) == 1)[0]
            mu0, var0 = self.stats(group0_train, group0_test)
            mu1, var1 = self.stats(group1_train, group1_test)
            size1 = len(group1_test)
            size0 = len(group0_test)
            erbt = (mu1 - mu0) / np.sqrt(var1 / size1 + var0 / size0)
            dof = (var1 / size1 + var0 / size0)**2/((var1 / size1)**2/(size1-1) + (var0 / size0)**2/(size0-1))
            dof = round(dof)
        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(s_train)):
                for j in range(len(s_train)):
                    if np.array(s_train)[i] - np.array(s_train)[j] > 0:
                        n += 1
                        bias_diff += self.delta_train[i]-self.delta_train[j]
            mean_train = bias_diff / n

            bias_diff = 0.0
            n = 0
            for i in range(len(s_test)):
                for j in range(len(s_test)):
                    if np.array(s_test)[i] - np.array(s_test)[j] > 0:
                        n += 1
                        bias_diff += self.delta_test[i] - self.delta_test[j]
            mean_test = bias_diff / n

            varA = np.var(self.delta_train, ddof=1)+np.var(self.delta_test, ddof=1)
            erbt = (mean_test - mean_train) / np.sqrt(varA*(1.0/len(s_test)))
            dof = len(s_test)-1
        p = t.sf(np.abs(erbt), dof)
        return p

    def RBD(self, s_train, s_test):
        if len(np.unique(s_train)) == 2 and len(np.unique(s_test)) == 2:
            group0_train = np.where(np.array(s_train) == 0)[0]
            group0_test = np.where(np.array(s_test) == 0)[0]
            group1_train = np.where(np.array(s_train) == 1)[0]
            group1_test = np.where(np.array(s_test) == 1)[0]
            mu0, var0 = self.stats(group0_train, group0_test)
            mu1, var1 = self.stats(group1_train, group1_test)
            size1 = len(group1_test)
            size0 = len(group0_test)
            # # varA = (var1  + var0) / 2
            # varB = (var1 * (len(group1_test) + len(group1_train) - 2) + var0 * (
            #                 len(group0_test) + len(group0_train) - 2)) / (
            #                 len(group1_test) + len(group1_train) + len(group0_test) + len(group0_train) - 4)
            # varC = (var1 * (len(group1_test)) * (len(group1_train)) + var0 * (len(group0_test)) * (
            #     len(group0_train))) / (
            #                (len(group1_test)) * (len(group1_train)) + (len(group0_test)) * (
            #            len(group0_train)))
            erbd = (mu1 - mu0) / np.sqrt((var0*(size0-1)+var1*(size1-1))/(size0+size1-2))

        else:
            bias_diff = 0.0
            n = 0
            for i in range(len(s_train)):
                for j in range(len(s_train)):
                    if np.array(s_train)[i] - np.array(s_train)[j] > 0:
                        n += 1
                        bias_diff += self.delta_train[i] - self.delta_train[j]
            mean_train = bias_diff / n

            bias_diff = 0.0
            n = 0
            for i in range(len(s_test)):
                for j in range(len(s_test)):
                    if np.array(s_test)[i] - np.array(s_test)[j] > 0:
                        n += 1
                        bias_diff += self.delta_test[i] - self.delta_test[j]
            mean_test = bias_diff / n

            mu_train, var_train = self.norm_stats(self.delta_train)
            mu_test, var_test = self.norm_stats(self.delta_test)
            varA = var_train + var_test
            erbd = (mean_test - mean_train) / np.sqrt(varA)
            # erbd = (mean_test - mean_train) / np.sqrt((var_train * (len(s_train)) + var_test * (len(s_test))) / (len(s_train)+len(s_test)))
        return erbd
