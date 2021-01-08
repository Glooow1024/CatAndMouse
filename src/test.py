#!/bin/env/python
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as c
from CatEnv import CatAndMouseEnv


if __name__=="__main__":
    model = np.load('qlearning_1_1.npz')
    print(model['map'])
    print(model['Q'])
    print(model['mode_obstacle'])
    print(model['mode_mouse'])