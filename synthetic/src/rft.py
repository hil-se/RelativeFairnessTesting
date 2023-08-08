from scipy import stats
import numpy as np
from data_reader import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from metrics import Metrics
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from biased_bridge import BiasedBridge
from pdb import set_trace

class RelativeFairnessTesting():

    def __init__(self, data="Community", regressor="Logistic", inject = None):
        datasets = {"Adult": load_adult, "German": load_german, "Bank": load_bank, "Default": load_default,
                    "Heart": load_heart, "Compas": load_compas, "StudentMat": load_student_mat,
                    "StudentPor": load_student_por}
        regressors = {"Linear": LinearRegression(positive=True),
                      "Logistic": LogisticRegression(max_iter=100000)}
        self.X, self.y, self.protected = datasets[data]()
        self.regressor = regressors[regressor]
        self.inject = inject
        self.preprocessor = None

    def t_str(self, p):
        return "%.2f : " %p
        # if p >=0 and p<=0.05:
        #     # Significant and positive
        #     t_color = "\cellcolor{green!20} "
        # elif p <0 and p>=-0.05:
        #     # Significant and negative
        #     t_color = "\cellcolor{red!20} "
        # else:
        #     # Not significant
        #     t_color = ""
        # return t_color

    def run(self):
        self.train_test_split()
        self.preprocess(self.X_train)
        X_train = self.preprocessor.transform(self.X_train)

        if self.inject is None:
            y_train = self.y_train
        else:
            y_train = self.inject_bias(self.X_train, self.y_train)

        if self.inject is None:
            y_test = self.y_test
        else:
            y_test = self.inject_bias(self.X_test, self.y_test)

        self.regressor.fit(X_train, y_train)
        pred_train = self.prediction(self.X_train)
        pred_test = self.prediction(self.X_test)
        result = {}

        m = Metrics(y_train, pred_train)
        result["Accuracy"] = 1.0 - m.mae()
        for key in self.protected:
            t_color = self.t_str(m.RBT(np.array(self.X_train[key])))
            result["RBD_PredTrain_" + str(key)] = t_color + "%.2f" %m.RBD(np.array(self.X_train[key]))

        m = Metrics(self.y_train, y_train)
        for key in self.protected:
            t_color = self.t_str(m.RBT(np.array(self.X_train[key])))
            result["RBD_GT_Train_" + str(key)] = t_color + "%.2f" %m.RBD(np.array(self.X_train[key]))

        m = Metrics(self.y_test, y_test)
        for key in self.protected:
            t_color = self.t_str(m.RBT(np.array(self.X_test[key])))
            result["RBD_GT_Test_" + str(key)] = t_color + "%.2f" %m.RBD(np.array(self.X_test[key]))

        m = Metrics(self.y_test, pred_test)
        for key in self.protected:
            t_color = self.t_str(m.RBT(np.array(self.X_test[key])))
            result["RBD_Unbiased_" + str(key)] = t_color + "%.2f" %m.RBD(np.array(self.X_test[key]))

        m = BiasedBridge(pred_train - y_train, pred_test - self.y_test)
        for key in self.protected:
            t_color = self.t_str(m.RBT(np.array(self.X_train[key]), np.array(self.X_test[key])))
            result["RBD_Biased_" + str(key)] = t_color + "%.2f" %m.RBD(np.array(self.X_train[key]), np.array(self.X_test[key]))

        return result

    def inject_bias(self, X_train, y_train):
        y_sigma = np.std(y_train)
        y_new = y_train[:]
        for a in self.inject:
            s = stats.zscore(X_train[a])
            # s = (X_train[a]-0.5)*2
            y_new = np.random.normal(y_new + s * self.inject[a] * y_sigma, np.abs(self.inject[a]) * y_sigma)
        if len(np.unique(y_train))==2:
            # Binary Classification
            y_new = np.array([1 if np.random.random()<y else 0 for y in y_new])
        return y_new


    def preprocess(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)])
        self.preprocessor.fit(X)

    def prediction(self, X):
        X_test = self.preprocessor.transform(X)
        y_pred = self.regressor.predict_proba(X_test)[:, 1]
        return y_pred

    def random_split(self, test_size=0.3):
        n = len(self.y)
        train = list(np.random.choice(n, int(n * (1-test_size)), replace=False))
        test = list(set(range(n)) - set(train))
        self.X_train = self.X.iloc[train]
        self.X_test = self.X.iloc[test]
        self.y_train = self.y[train]
        self.y_test = self.y[test]
        self.X_train.index = range(len(self.X_train))
        self.X_test.index = range(len(self.X_test))

    def train_test_split(self, test_size=0.3):
        # Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(self.y)):
            key = tuple([self.X[a][i] for a in self.protected] + [self.y[i]])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        train = []
        test = []
        for key in groups:
            testing = list(np.random.choice(groups[key], int(len(groups[key])*test_size), replace=False))
            training = list(set(groups[key]) - set(testing))
            test.extend(testing)
            train.extend(training)
        self.X_train = self.X.iloc[train]
        self.X_test = self.X.iloc[test]
        self.y_train = self.y[train]
        self.y_test = self.y[test]
        self.X_train.index = range(len(self.X_train))
        self.X_test.index = range(len(self.X_test))

