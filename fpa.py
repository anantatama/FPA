import csv
import numpy as np
import pandas as pd
from mantegna import *

with open("winequality-white.csv", 'r') as f:
    wines = list(csv.reader(f, delimiter=";"))

data = np.array(wines[1:], dtype=np.float)
print(data)

#data = pd.read_csv("winequality-white.csv", header=0)
#print(data.head(0))

def model():
    """
    Disini model dari Artificial Neural Networknya
    """

def init_pop():
    """
    Fungsi untuk meng-generate N buah individu dalam populasi
    """
    individu = np.zeros((jumlah_individu, jumlah_variabel))
    for i in range(jumlah_individu):
        for j in range(jumlah_variabel):
            individu[i][j] = np.random.uniform(0,1)
    return individu

def find_best():
    """
    Mencari g*, yaitu individu terbaik pada satu generasi(iterasi)
    """
    pass



def FPA():
    init_pop()
    find_best()
    p = 0.8
    for i in range(len(individu)):
        rand = np.random.uniform(0,1)
        if rand < p:
            L = stepsize(beta)
            GP = globalPollination()
        else:
            Eps = np.random.uniform(0.1)
            LP = localPollination()
        evaluate()
        update_solution()
    find_best()
    if t >= 5:
        delta_fg = calc_delta_fg()
    else:
        delta_fg = 1

    while(delta_fg > 10**-5):
        for i in range(len(individu)):
            rand = np.random.uniform(0,1)
            if rand < p:
                L = stepsize(beta)
                GP = globalPollination()
            else:
                Eps = np.random.uniform(0.1)
                LP = localPollination()
            evaluate()
            update_solution()
        find_best()
        if t >= 5:
            delta_fg = calc_delta_fg()
        else:
            delta_fg = 1


