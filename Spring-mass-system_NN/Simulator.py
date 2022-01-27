#!/usr/bin/env python
# coding: utf-8

# In[7]:


#### This is a simulation file ####
#### and meant to be imported  ####


import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# In[8]:


### Function that returns dx/dt ###
def mydiff(t, x, c, k, m, F):
    
    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*x[0])/m
    dxdt = [dx1dt, dx2dt]
    return dxdt


# In[9]:


## Solve using ODEint, If used swap mydiff t and x args ###
#x = odeint(mydiff, x_init, t)
#x1 = x[:,0]
#x2 = x[:,1]

### Solve using IVP ###
def generate(c,k,m,F, x0=0): ## Generate function which can be called to create dataset

    ### Initialization ###
    tstart = 0
    tstop = 60
    increment = 1000

    ### Initial condition ###
    x_init = [x0,0]
    #t = np.arange(tstart,tstop+1,increment)
    t= np.linspace(tstart, tstop, increment)

    argss = (c, k, m, F) ## Args for solving
    
    
    x = solve_ivp(mydiff, [tstart, tstop], x_init, args=argss, method='RK45', t_eval=t)
    x1 = x.y[0]
    v1 = x.y[1]
    return [t, v1, x1] ## Returning dataset


# In[10]:


### Plot the Results ###


#plt.plot(x.t, x1)
#plt.plot(x.t, v1)
#plt.xlabel('t')
#plt.legend(['position', 'speed'], shadow=True)
#plt.title('Spring mass system System')
#plt.show()

