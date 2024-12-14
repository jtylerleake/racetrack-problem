# -*- coding: utf-8 -*-
"""
contains functions used throughout the program for various
purposes (primarily for training)

@name:          utils.py
@author:        J. Tyler Leake
@last update:   08-19-2024
"""

import numpy as np
import random



def init_q_table(env):
    '''
    initializes the action value table for the algorithm 
    with all states set to random values
    '''
    X_cord_dim = env.X_cord_dim
    Y_cord_dim = env.Y_cord_dim
    X_velo_dim = abs(env.X_velo_dim[1] - env.X_velo_dim[0])
    Y_velo_dim = abs(env.Y_velo_dim[1] - env.Y_velo_dim[0])
    actions_dim = len(env.actions)
    
    q_table = np.zeros((X_cord_dim, Y_cord_dim, X_velo_dim, Y_velo_dim, actions_dim))
    
    for X_cord in range(X_cord_dim):
        for Y_cord in range(Y_cord_dim):
            for X_velo in range(X_velo_dim):
                for Y_velo in range(Y_velo_dim):
                    q_table[X_cord, Y_cord, X_velo, Y_velo, :] = \
                        np.random.rand(actions_dim)

    return q_table
    

def epsilon_greedy(car, q_vals, p_explore):
    '''
    implements the epsilon-greedy strategy for selecting an 
    explore vs. exploit action in the env.
    '''
    # randomly sample from uniform distribution; if sample is less than 
    # exploration threshold, explore. else exploit (q_table argmax)
    
    if np.random.uniform(0, 1) < p_explore:            
        action = random.choice(car.env.actions)
        action_idx = car.env.actions.index(action)
        q_value = q_vals[action_idx]
        
    else: 
        action_idx = np.argmax(q_vals)
        action = car.env.actions[action_idx]
        q_value = q_vals[action_idx]
        
    return action, action_idx, q_value
