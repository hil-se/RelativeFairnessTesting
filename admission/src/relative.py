import pandas as pd
import numpy as np
from metrics import Metrics
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVR
from biased_bridge import BiasedBridge


class RelativeFairness:
    def __init__(self, data_path="../data/phd_2023.csv", data_path2="../data/phd_2022.csv", targets=["Eval 1", "Eval 2", "Eval 3"], protected_attrs = ["US Degree"]):
        self.data = pd.read_csv(data_path)
        self.data2 = pd.read_csv(data_path2)
        self.data = self.handlenan(self.data)
        self.data2 = self.handlenan(self.data2)
        self.data = self.english(self.data)
        self.data2 = self.english(self.data2)
        self.data = self.GRE(self.data)
        self.data2 = self.GRE(self.data2)
        self.independent = ["Faculty Identified", "Institution Region (BS)", "Institution Ranking (BS)",
                            "BS GPA", "Institution Region (MS)", "Institution Ranking (MS)", "MS GPA",
                            "Research Experience", "Int'l pubs", "GRE Quan", "GRE Ver", "GRE AWA", "GRE Total",
                            "English", "Has MS", "US Degree"]
        self.X = self.data[self.independent]
        self.X2 = self.data2[self.independent]
        self.targets = targets
        self.protected = protected_attrs
        self.clf = SVR(kernel="rbf", C=1.5, gamma='auto', epsilon=0.1, shrinking=True)

    def handlenan(self, X):
        for i in range(len(X)):
            if pd.isna(X["Institution Region (BS)"][i]):
                X["Institution Region (BS)"][i] = "None"
            if pd.isna(X["Institution Region (MS)"][i]):
                X["Institution Region (MS)"][i] = "None"
            if pd.isna(X["Institution Ranking (BS)"][i]):
                X["Institution Ranking (BS)"][i] = 1000
            if pd.isna(X["Institution Ranking (MS)"][i]):
                X["Institution Ranking (MS)"][i] = 1000
            if pd.isna(X["BS GPA"][i]):
                X["BS GPA"][i] = 0.6
            if pd.isna(X["MS GPA"][i]):
                X["MS GPA"][i] = 0.6
            if pd.isna(X["GRE Quan"][i]):
                X["GRE Quan"][i] = 150
            if pd.isna(X["GRE Ver"][i]):
                X["GRE Ver"][i] = 150
            if pd.isna(X["GRE AWA"][i]):
                X["GRE AWA"][i] = 3.0
            if pd.isna(X["GRE Total"][i]):
                X["GRE Total"][i] = 300
        return X

    def english(self, data):
        english_skill = []
        for i in range(len(data)):
            if pd.isna(data["TOEFL"][i]):
                if pd.isna(data["IELTS"][i]):
                    english_skill.append(0.6)
                else:
                    english_skill.append(data["IELTS"][i] / 9.0)
            else:
                if pd.isna(data["IELTS"][i]):
                    english_skill.append(data["TOEFL"][i] / 120.0)
                else:
                    english_skill.append(max([data["TOEFL"][i] / 120.0, data["IELTS"][i] / 9.0]))

        data["English"] = english_skill
        return data

    def GRE(self, data):
        gre_quan = []
        gre_ver = []
        gre_awa = []
        gre_total = []
        for i in range(len(data)):
            if pd.isna(data["GRE Quan"][i]):
                gre_quan.append(0.5)
            else:
                gre_quan.append((data["GRE Quan"][i] - 130)/40.0)
            if pd.isna(data["GRE Ver"][i]):
                gre_ver.append(0.5)
            else:
                gre_ver.append((data["GRE Ver"][i] - 130)/40.0)
            if pd.isna(data["GRE AWA"][i]):
                gre_awa.append(0.5)
            else:
                gre_awa.append((data["GRE AWA"][i])/6.0)
            gre_total.append((gre_ver[i]+gre_quan[i])/2.0)
        data["GRE Quan"] = gre_quan
        data["GRE Ver"] = gre_ver
        data["GRE AWA"] = gre_awa
        data["GRE Total"] = gre_total
        return data

    def preprocess(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)])

    def fit(self, X_train, y_train):
        self.preprocess(X_train)
        X = self.preprocessor.fit_transform(X_train)
        self.clf.fit(X, y_train)

    def predict(self, X_test):
        X = self.preprocessor.transform(X_test)
        return self.clf.predict(X)

    def run(self):
        for base in self.targets:
            results = []
            y_train = np.array(self.data[base])
            self.fit(self.X, y_train)
            preds = self.predict(self.X2)
            preds_train = self.predict(self.X)

            for target in self.targets:
                # GT on training set
                result = {"Pair": base+"/"+target, "Metric": "GT Train"}
                m = Metrics(self.data[target], self.data[base])
                result["MAE"] = "%.2f" % m.mae()
                for A in self.protected:
                    result[A] = "(%.2f) %.2f" % (m.RBT(self.data[A]), m.RBD(self.data[A]))
                results.append(result)
                # GT on test set
                result = {"Pair": base+"/"+target, "Metric": "GT Test"}
                m = Metrics(self.data2[target], self.data2[base])
                result["MAE"] = "%.2f" % m.mae()
                for A in self.protected:
                    result[A] = "(%.2f) %.2f" % (m.RBT(self.data2[A]), m.RBD(self.data2[A]))
                results.append(result)
                # Prediction on test set
                result = {"Pair": base + "/" + target, "Metric": "Unbiased Bridge"}
                m = Metrics(self.data2[target], preds)
                result["MAE"] = "%.2f" % m.mae()
                for A in self.protected:
                    result[A] = "(%.2f) %.2f" % (m.RBT(self.data2[A]), m.RBD(self.data2[A]))
                results.append(result)
                # predict test
                result = {"Pair": base + "/" + target, "Metric": "Biased Bridge"}
                m = BiasedBridge(preds_train - self.data[base], preds - self.data2[target])
                result["MAE"] = "%.2f" % 0.0
                for A in self.protected:
                    result[A] = "(%.2f) %.2f" % (
                        m.RBT(self.data[A], self.data2[A]), m.RBD(self.data[A], self.data2[A]))
                results.append(result)

            df = pd.DataFrame(results)
            df.to_csv("../result/result_" + base + ".csv", index=False)
            df = ''
        return results