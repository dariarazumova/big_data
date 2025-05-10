# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:12:23 2019

@author: user
"""
import numpy as np
from matplotlib import pyplot as plt

r_mesh = np.arange(0,1,0.01)
#Dx  = (1+r_mesh)/2

#fig,ax = plt.subplots()
#ax.plot(r_mesh, Dx)

Dx  = (1 - r_mesh)*(5*r_mesh+3)/8/(r_mesh+1)

fig,ax = plt.subplots()
ax.plot(r_mesh, Dx)
