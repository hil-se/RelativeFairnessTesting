from relative_class import RelativeFairness
import numpy as np

if __name__ == "__main__":
    np.random.seed(0)
    r = RelativeFairness(data_path="../data/phd_2023.csv", data_path2="../data/phd_2022.csv", targets=["Eval 1", "Eval 2", "Eval 3"])
    r.run()