import matplotlib.pyplot as plt
from data_reader import load_scut
import numpy as np
from pdb import set_trace
import pandas as pd

def plotout(name):
    A = "race"
    df = pd.read_csv("../outputs/"+name+".csv")
    base_target0 = df["base"][df[A] == 0]-df["target"][df[A] == 0]
    base_target1 = df["base"][df[A] == 1] - df["target"][df[A] == 1]
    delta_train0 = df["pred"][(df["split"]==1) & (df[A] == 0)] - df["base"][(df["split"]==1) & (df[A] == 0)]
    delta_train1 = df["pred"][(df["split"]==1) & (df[A] == 1)] - df["base"][(df["split"]==1) & (df[A] == 1)]
    delta_test0 = df["pred"][(df["split"]==0) & (df[A] == 0)] - df["target"][(df["split"]==0) & (df[A] == 0)]
    delta_test1 = df["pred"][(df["split"]==0) & (df[A] == 1)] - df["target"][(df["split"]==0) & (df[A] == 1)]
    ####
    fig, axs = plt.subplots(2, 3, sharey=True, tight_layout=True)
    axs[0, 0].hist(base_target0)
    axs[0, 1].hist(delta_train0)
    axs[0, 2].hist(delta_test0)
    axs[1, 0].hist(base_target1)
    axs[1, 1].hist(delta_train1)
    axs[1, 2].hist(delta_test1)
    plt.savefig("../outputs/figs/"+name+".png")
    plt.close()

def plotreal(A='sex'):
    data, protected = load_scut()
    files = ["P1", "P2", "P3", "Average"]
    for first in range(3):
        for second in range(1,4):
            error = data[files[first]]-data[files[second]]
            error0 = error[np.where(data[A] == 0)[0]]
            error1 = error[np.where(data[A] == 1)[0]]
            fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
            axs[0].hist(error0)
            axs[1].hist(error1)
            plt.savefig("../outputs/figs/" + files[first]+"-" +files[second]+ ".png")
            plt.close()

def run():
    files = ["P1", "P2", "P3", "Average"]
    names = [f1+"_"+f2 for f2 in files for f1 in files]
    for name in names:
        plotout(name)

if __name__ == "__main__":
    # plotreal("race")
    run()