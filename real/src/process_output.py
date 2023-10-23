import numpy as np
from metrics import Metrics
import pandas as pd
from bridge import Bridge
from biased_bridge import BiasedBridge
from pdb import set_trace


def run(base="P1"):
    cols = ["P1", "P2", "P3", "Average"]
    protected = ["sex", "race"]
    results = []
    for target in cols:
        data = pd.read_csv("../outputs/"+base+"_"+target+".csv")
        train = data['split'] == 1
        val = data['split'] == 2
        test = data['split'] == 0
        training = data['split'] > 0

        # result = {"Pair": base, "Metric": "Train"}
        # m = Metrics(data["base"][train], data['pred'][train])
        # result["MAE"] = m.mae()
        # for A in protected:
        #     result[A] = "(%.2f) %.2f" % (m.RBT(data[A][train]), m.RBD(data[A][train]))
        # results.append(result)


        # GT on training set
        result = {"Pair": base + "/" + target, "Metric": "GT Train"}
        m = Metrics(data["target"][training], data["base"][training])
        result["MAE"] = m.mae()
        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][training]), m.RBD(data[A][training]))
        results.append(result)
        # GT on val set
        result = {"Pair": base + "/" + target, "Metric": "GT Val"}
        m = Metrics(data["target"][val], data["base"][val])
        result["MAE"] = m.mae()
        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][val]), m.RBD(data[A][val]))
        results.append(result)
        # GT on test set
        result = {"Pair": base + "/" + target, "Metric": "GT Test"}
        m = Metrics(data["target"][test], data["base"][test])
        result["MAE"] = m.mae()
        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][test]), m.RBD(data[A][test]))
        results.append(result)
        # Prediction on test set
        result = {"Pair": base + "/" + target, "Metric": "Unbiased Bridge"}
        m = Metrics(data["target"][test], data['pred'][test])
        result["MAE"] = m.mae()
        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][test]), m.RBD(data[A][test]))
        results.append(result)
        # predict test
        result = {"Pair": base + "/" + target, "Metric": "Biased Bridge"}
        m = BiasedBridge(data['pred'][val] - data["base"][val], data['pred'][test] - data["target"][test])
        result["MAE"] = 0.0
        for A in protected:
            result[A] = "(%.2f) %.2f" % (m.RBT(data[A][val], data[A][test]), m.RBD(data[A][val], data[A][test]))
        results.append(result)
        # predict test bridge
        # result = {"Pair": base + "/" + target, "Metric": "Bridge"}
        # m = Bridge(data, protected)
        # result["MAE"] = 0.0
        # for A in protected:
        #     result[A] = "(%.2f) %.2f" % (
        #     m.RBT(A), m.RBD(A))
        # results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("../results/output_result_" + base + ".csv", index=False)

if __name__ == "__main__":
    cols = ["P1", "P2", "P3", "Average"]
    for base in cols:
        run(base=base)
