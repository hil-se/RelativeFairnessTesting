import numpy as np
from data_reader_class import load_scut
from metrics import Metrics
import pandas as pd
from vgg_pre_class import VGG_Pre
from biased_bridge import BiasedBridge

class RelativeFairnessTesting():

    def __init__(self, rating_cols = ["P1", "P2", "P3", "Average"]):
        self.rating_cols = rating_cols
        self.data, self.protected = load_scut(rating_cols = rating_cols)
        self.features = np.array([pixel for pixel in self.data['pixels']])/255.0

    def run(self):
        n = len(self.data)
        test = list(np.random.choice(n, int(n * 0.4), replace=False))
        train = list(set(range(n)) - set(test))
        val = list(np.random.choice(test, int(n * 0.2), replace=False))
        test = list(set(test) - set(val))
        training = list(set(train) | set(val))

        X_train = self.features[train]
        X_val = self.features[val]


        for base in self.rating_cols:
            results = []
            y_train = np.array(self.data[base][train])
            y_val = np.array(self.data[base][val])
            self.learn(X_train, y_train, X_val, y_val)
            preds = self.model.predict(self.features)

            m = Metrics(self.data[base][train], preds[train])
            result = {"Pair": base, "Metric": "Train"}
            result["Accuracy"] = "%.2f" %(1.0 - m.mae())
            for A in self.protected:
                result[A] = "(%.2f) %.2f" % (m.RBT(self.data[A][train]), m.RBD(self.data[A][train]))
            results.append(result)

            for target in self.rating_cols:
                # GT on training set
                result = {"Pair": base+"/"+target, "Metric": "GT Train"}
                m = Metrics(self.data[target][training], self.data[base][training])
                result["Accuracy"] = "%.2f" %(1.0 - m.mae())
                for A in self.protected:
                    result[A] = "(%.2f) %.2f" % (m.RBT(self.data[A][training]), m.RBD(self.data[A][training]))
                results.append(result)
                # GT on test set
                result = {"Pair": base+"/"+target, "Metric": "GT Test"}
                m = Metrics(self.data[target][test], self.data[base][test])
                result["Accuracy"] = "%.2f" %(1.0 - m.mae())
                for A in self.protected:
                    result[A] = "(%.2f) %.2f" % (m.RBT(self.data[A][test]), m.RBD(self.data[A][test]))
                results.append(result)
                # Prediction on test set
                result = {"Pair": base + "/" + target, "Metric": "Unbiased Bridge"}
                m = Metrics(self.data[target][test], preds[test])
                result["Accuracy"] = "%.2f" %(1.0 - m.mae())
                for A in self.protected:
                    result[A] = "(%.2f) %.2f" % (m.RBT(self.data[A][test]), m.RBD(self.data[A][test]))
                results.append(result)
                # predict test
                result = {"Pair": base + "/" + target, "Metric": "Biased Bridge"}
                m = BiasedBridge(preds[val] - self.data[base][val], preds[test] - self.data[target][test].to_numpy())
                result["Accuracy"] = "%.2f" %(1.0)
                for A in self.protected:
                    result[A] = "(%.2f) %.2f" % (
                    m.RBT(self.data[A][val], self.data[A][test]), m.RBD(self.data[A][val], self.data[A][test]))
                results.append(result)
            df = pd.DataFrame(results)
            df.to_csv("../results/result_class_" + base + ".csv", index=False)
            df = ''
        return results


    def learn(self, X, y, X_val, y_val):
        # train a model on the training set and use the model to predict on the test set
        self.model = VGG_Pre()
        self.model.fit(X, y, X_val, y_val)

