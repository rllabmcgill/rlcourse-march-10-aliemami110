# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 14:49:21 2017

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 15:01:54 2017

@author: user
"""

import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

import plotly
plotly.tools.set_credentials_file(username='emamiali', api_key='2ctBeK6YDXLcRyB06Ipf')
import plotly.plotly as py
import plotly.graph_objs as go

def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """

    ZeroList=[0]*9
    OneList=[0]*9
    TwoList=[0]*9
    
    for i in range(0,len(V)):
        x=i%3
        y=i/3
        
        if x==0:
            ZeroList[y]=V[i]
        if x==1:
            OneList[y]=V[i]
        if x==2:
            TwoList[y]=V[i]

    zYEA=[ZeroList,OneList,TwoList]
    #z=[[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8]]
    
    data = [
    go.Heatmap(
        z=zYEA,
        x=['0', '1', '2'],
        y=['0','1','2','3','4','5','6','7','8']
    )
]

    py.iplot(data, filename='labelled-heatmap')

 
    


