# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:19:59 2017

@author: user
"""

import numpy as np
from gym.envs.toy_text import discrete

FL = 0 
DF = 1
FR = 2

class WallFollowingEnv(discrete.DiscreteEnv):
    
    """
    
    T T T 
    o x o
    o o o
    o o o
    
    States 0:24 are the states in the game. We can find the location number using:
    index mod 8 is the row number, mod 3 is the column number. 
    Last three states are 24:26 and they represent wall, open space and exit, respectively.
    
    You can take actions in each direction (FL=0, DF=1, FR=2).
    Actions moving you too far from the wall (three away) lead to termination, reward -1.
    Hitting a wall is -1. You recieve a reward of +1 for finishing.
    """
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, shape=[9,3]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
             raise ValueError('shape argument must be a list/tuple of length 2')
             
        self.shape = shape
        
        nS = np.prod(shape)
        nA = 3
        
        self.policy=np.ones([nS, nA]) / nA
        
        MAX_Y= shape[0]-1
        MAX_X= shape[1]

        P={}
        grid=np.arange(nS).reshape(shape)
        it = np.nditer(grid,flags=['multi_index'])
        
        while not it.finished:
            s = it.iterindex 
            y, x = it.multi_index
            

            if x==0:
                self.policy[s][FL]=1.0/6 
                self.policy[s][DF]=1/3.0 
                self.policy[s][FR]=1/2.0
            if x==1:
                self.policy[s][FL]=1/4.0
                self.policy[s][DF]=1/2.0
                self.policy[s][FR]=1/4.0
            if x==2:
                self.policy[s][FL]=1/2.0 
                self.policy[s][DF]=1/3.0 
                self.policy[s][FR]=1/6.0


            P[s] = {a : [] for a in range(nA)}

            accidentalWall=0.0
            accidentalOpen=0.0
            obstacleHit=0.0

            is_fail = lambda s: s == nS-3 or s ==nS-2 
            is_succeed = lambda s: s == nS-1 
            is_done= lambda s: is_fail(s) or is_succeed(s) or obstacleHit==1.0
            
            if is_fail(s):
                reward=-1
            elif is_succeed(s):
                reward=1
            else:
                if x==0:
                    reward=-0.1
                    #accidentalWall=np.random.choice([0.0,1.0], p=[0.2,0.8])
                    #if accidentalWall==1.0:             
                        #reward=-1
                        
                elif x==1:
                    reward=0
                    if y==5:
                        obstacleHit=1.0
                        reward=-1
                elif x==2:
                    reward=-0.2
                    #accidentalOpen=np.random.choice([0.0,1.0], p=[0.3,0.7])
                    #if accidentalOpen==1.0:
                        #reward=-1
                
                        
            


            # We're stuck in a terminal state
            if is_done(s):
                P[s][FL] = [(1.0, s, reward, True)]
                P[s][DF] = [(1.0, s, reward, True)]
                P[s][FR] = [(1.0, s, reward, True)]

            # Not a terminal state
            else:    
                if y==(MAX_Y-1):
                    ns_forward_left=nS-1
                    ns_directly_forward=nS-1
                    ns_forward_right=nS-1
                else:
                    ns_forward_left = nS-3 if x == 0 else s + shape[1]-1
                    ns_directly_forward = s + shape[1]
                    ns_forward_right = nS-2 if x == (MAX_X - 1) else s + shape[1]+1


                P[s][FL] = [(1.0, ns_forward_left, reward, is_done(ns_forward_left))]
                P[s][DF] = [(1.0, ns_directly_forward, reward, is_done(ns_directly_forward))]
                P[s][FR] = [(1.0, ns_forward_right, reward, is_done(ns_forward_right))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS
        self.P = P


        super(WallFollowingEnv, self).__init__(nS, nA, P, isd)

        