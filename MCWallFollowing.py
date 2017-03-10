# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 12:45:02 2017

@author: user
"""

import matplotlib
import numpy as np
import sys
import matplotlib.pyplot as plt

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from wallfollowing import WallFollowingEnv
import plottingWallFollowing

matplotlib.style.use('ggplot')

env = WallFollowingEnv()

def epsilon_greedy_policy(Q, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
    
def mc_control(env, num_episodes, discount_factor=0.9, epsilon=0.1):
    
    #Dictionary of returns sum and count
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    #Dictionary mapping state to action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    #Create policy based on Q, epsilon, and number of actions
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i in range(1, num_episodes + 1):

        # Generate episode
        episode = []
        env.reset()
        state = np.random.choice([0,1,2])
        reward=0
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            prob,next_state, reward, done = env.P[state][action][0]
            episode.append((state, action, reward))
            if done:
                state=next_state
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                prob,next_state, reward, done = env.P[state][action][0]
                episode.append((state,action,reward))
                break
            state = next_state

        state_action_pair_in_episode = set([(x[0], x[1]) for x in episode])
        for state, action in  state_action_pair_in_episode:
            state_action_pair = (state, action)
            #First occurence of state action pair in episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            #All rewards since first occurence, discounted
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            
            #Average return calculated
            returns_sum[state_action_pair] += G
            returns_count[state_action_pair] += 1.0
            Q[state][action] = returns_sum[state_action_pair] / returns_count[state_action_pair]
        
        #Q dictionary changed implicitly and policy improved
    
    return Q, policy
  

index=0
Reward=['']*10
#episodeValues=[50000]
discountFactorValues=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for discount in discountFactorValues:
    
    Q, policy = mc_control(env, num_episodes=5000, epsilon=0.05, discount_factor=discount)
    
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    plottingWallFollowing.plot_value_function(V, title="Optimal Value Function")
    
    
    rewardBestArray = [0]*10000
    
    for i in range(0, 10000):
    
            episode = []
            env.reset()
            state = np.random.choice([0,1,2])
            totalReward=0
            for t in range(100):
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, _ = env.step(action)
                totalReward+=reward
                episode.append((state, action, reward))
                if done:
                    state=next_state
                    probs = policy(state)
                    action = np.random.choice(np.arange(len(probs)), p=probs)
                    prob,next_state, reward, done = env.P[state][action][0]
                    episode.append((state,action,reward))
                    totalReward+=reward
                    rewardBestArray[i]=totalReward
                    break
                state = next_state
    
    rewardBestArray=np.array(rewardBestArray)
    averageBestReward=np.mean(rewardBestArray)
    
    Reward[index]=averageBestReward
    index=index+1
    
    
    rewardSampleArray = [0]*10000
    
    def sample_policy(observation):
        policy = np.ones([env.nS, env.nA]) / env.nA
        policy[:,2]=0.0
        policy[:,1]=1.0
        policy[:,0]=0.0
        #policy=env.policy
        return policy[observation]
    
    
    for i in range(0, 10000):
    
            episode = []
            env.reset()
            state = np.random.choice([0,1,2])
            totalReward=0
            for t in range(100):
                probs = sample_policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                prob,next_state, reward, done= env.P[state][action][0]
                totalReward+=reward
                episode.append((state, action, reward))
                if done:
                    state=next_state
                    probs = sample_policy(state)
                    action = np.random.choice(np.arange(len(probs)), p=probs)
                    prob,next_state, reward, done = env.P[state][action][0]
                    episode.append((state,action,reward))
                    totalReward+=reward
                    rewardSampleArray[i]=totalReward
                    break
                state = next_state
    
    rewardSampleArray=np.array(rewardSampleArray)
    averageSampleReward=np.mean(rewardSampleArray)
    
plt.plot(discountFactorValues, Reward, '-o')
plt.axis([0,1,-0.5,0.5])
plt.show()
    
        
    
