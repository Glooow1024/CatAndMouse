#!/bin/env/python
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as c
from CatEnv import CatAndMouseEnv
from Qlearning import Qlearning
from sarsa import SARSA


if __name__=="__main__":
    '''method = 0
    if method==0:
        model = np.load('qlearning_1_1.npz')
        env = CatAndMouseEnv(mode_obstacle=model['mode_obstacle'],mode_mouse=model['mode_mouse'],map=model['map'],mouse=model['mouse'])
        q = Qlearning(mode_obstacle=model['mode_obstacle'],mode_mouse=model['mode_mouse'],map=model['map'],Q=model['Q'],mouse=model['mouse'])
    else:
        model = np.load('sarsa_1_1.npz')
        env = CatAndMouseEnv(mode_obstacle=model['mode_obstacle'],mode_mouse=model['mode_mouse'],map=model['map'],mouse=model['mouse'])
        q = SARSA(mode_obstacle=model['mode_obstacle'],mode_mouse=model['mode_mouse'],map=model['map'],Q=model['Q'],mouse=model['mouse'])
    q.visualization()
    print(model['map'])
    print(model['Q'])
    print(model['mode_obstacle'])
    print(model['mode_mouse'])'''

    method = 0
    if method==0:
        model = np.load('qlearning_1_1.npz')
        q = Qlearning(8,8,mode_obstacle=1,mode_mouse=1,Q=model['Q'])
    else:
        model = np.load('sarsa_1_1.npz')
        q = SARSA(8,8,mode_obstacle=1,mode_mouse=1,Q=model['Q'])
    q.visualization()
    q.test()
    
    print("env closed")
