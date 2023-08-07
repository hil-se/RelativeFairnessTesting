import numpy as np
from data_reader import load_scut
from metrics import Metrics
import pandas as pd
from vgg_pre import VGG_Pre
from biased_bridge import BiasedBridge

class RelativeFairnessTesting():

    def __init__(self):
        self.data, self.protected = load_scut()
        self.features = np.array([pixel for pixel in self.data['pixels']])/255.0

    def t_str(self, p):
        if p >=0 and p<=0.05:
            # Significant and positive
            t_color = "\cellcolor{green!20} "
        elif p <0 and p>=-0.05:
            # Significant and negative
            t_color = "\cellcolor{red!20} "
        else:
            # Not significant
            t_color = ""
        return t_color


    def run(self):
        n = len(self.data)
        train = list(np.random.choice(n, int(n*0.7), replace=False))
        test = list(set(range(n)) - set(train))
        # train, test = self.train_test_split(test_size=0.3)


        cols = ["P1", "P2", "P3", "Average"]

        X_train = self.features[train]
        X_test = self.features[test]

        for base in cols:
            results = []
            y_train = np.array(self.data[base][train])
            predicts, pred_train = self.learn(X_train, y_train, X_test)

            m = Metrics(self.data[base][train], pred_train)
            result = {"Pair": base, "Metric": "Train"}
            result["Accuracy"] = 1.0 - m.mae()
            for A in self.protected:
                t_color = self.t_str(m.RBT(self.data[A][train]))
                result[A+": "+"RBD"] = t_color + "%.2f" % m.RBD(self.data[A][train])
            results.append(result)

            for target in cols:
                # GT on training set
                result = {"Pair": base+"/"+target, "Metric": "GT Train"}
                m = Metrics(self.data[target][train], self.data[base][train])
                result["Accuracy"] = 1.0 - m.mae()
                for A in self.protected:
                    t_color = self.t_str(m.RBT(self.data[A][train]))
                    result[A+": "+"RBD"] = t_color + "%.2f" %m.RBD(self.data[A][train])
                results.append(result)
                # GT on test set
                result = {"Pair": base+"/"+target, "Metric": "GT Test"}
                m = Metrics(self.data[target][test], self.data[base][test])
                result["Accuracy"] = 1.0 - m.mae()
                for A in self.protected:
                    t_color = self.t_str(m.RBT(self.data[A][test]))
                    result[A + ": " + "RBD"] = t_color + "%.2f" %m.RBD(self.data[A][test])
                results.append(result)
                # Prediction on test set
                result = {"Pair": base + "/" + target, "Metric": "Unbiased Bridge"}
                m = Metrics(self.data[target][test], predicts)
                result["Accuracy"] = 1.0 - m.mae()
                for A in self.protected:
                    t_color = self.t_str(m.RBT(self.data[A][test]))
                    result[A + ": " + "RBD"] = t_color + "%.2f" %m.RBD(self.data[A][test])
                results.append(result)
                # predict test
                result = {"Pair": base + "/" + target, "Metric": "Biased Bridge"}
                m = BiasedBridge(pred_train - y_train, predicts - self.data[target][test].to_numpy())
                result["Accuracy"] = 1.0
                for A in self.protected:
                    t_color = self.t_str(m.RBT(self.data[A][train], self.data[A][test]))
                    result[A + ": " + "RBD"] = t_color + "%.2f" % m.RBD(self.data[A][train], self.data[A][test])
                results.append(result)


            df = pd.DataFrame(results)
            df.to_csv("../results/result_" + base + ".csv", index=False)
        return results

    def train_test_split(self, test_size=0.3):
        # Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(self.data)):
            key = tuple([self.data[a][i] for a in self.protected])
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
        return train, test

    def learn(self, X, y, X_test):
        # train a model on the training set and use the model to predict on the test set
        # model = VGG()
        self.model = VGG_Pre()
        self.model.fit(X, y)
        # preds = model.predict(X_test)
        preds = self.model.decision_function(X_test).flatten()
        preds_train = self.model.decision_function(X).flatten()
        # print(np.unique(preds))
        return preds, preds_train


