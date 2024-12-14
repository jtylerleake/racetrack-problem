# -*- coding: utf-8 -*-
"""
contains implementation of the Q-Learning algorithm for the 
racetrack problem as a class. 

@name:          QLearning.py
@author:        J. Tyler Leake
@last update:   08-19-2024
"""

import numpy as np
from Racetrack import *
from Car import *
from utils import *



class QLearning:
    
    def __init__(self, Car, r_learning, r_discount, r_decay, p_explore, 
                 episodes = 10, max_itr = 1000):

        self.car = Car  # agent; contains the racetrack env.
        self.q_table = None  # action-value function table
        
        # model hyperparameters
        self.r_learning = r_learning
        self.r_discount = r_discount
        self.r_decay = r_decay
        self.p_explore = p_explore
        self.episodes = episodes
        self.max_itr = max_itr
        
        # results
        self.training_results = {}
        
    def train(self):
        '''
        implementation of off-policy Q-learning algorithm using an
        epsilon-greedy explore vs. exploit strategy. 
        
        return: none; self.q_table updated directly
        '''
        # initialize action-value function (Q)
        self.q_table = init_q_table(self.car.env) 
        
        for episode in range(self.episodes): 
            
            self.car.restart_env() # restart the agent at starting line
            
            Y_cord = self.car.X_cord_cur # retrieve agent's state vals
            X_cord = self.car.Y_cord_cur
            X_velo = self.car.X_velo
            Y_velo = self.car.Y_velo
            
            # for each episode, iterate until either the agent 
            # reaches the finish line or 'max_itr' is hit
            
            ep_itr = 0
            done = False

            while not done:
                
                # retrieve the q-values for the current state
                q_vals = self.q_table[X_cord, Y_cord, X_velo, Y_velo]
                
                # action selection/transition probability: perform random 
                # experiment and either a) do nothing or b) select action
                
                rand_sample = np.random.uniform(0, 1)
                
                # action: do nothing
                if rand_sample > self.car.env.p_transition: 
                    action = (0, 0)
                    action_idx = self.car.env.actions.index(action)
                    q_val = self.q_table[X_cord, Y_cord, X_velo, Y_velo, action_idx]
                
                # action: apply explore vs. exploit strategy
                if rand_sample <= self.car.env.p_transition:
                    action, action_idx, q_val = epsilon_greedy(self.car, \
                                                          q_vals, self.p_explore)
                
                self.car.update_state(action) # perform action
                
                # check if car has finished. if true, then the episode 
                # has completed. else update the q-table
                
                if self.car.is_finished: 
                    done = True 
                    
                if not self.car.is_finished:
                    
                    # retrieve the new state values of the car
                    X_cord = self.car.X_cord_cur
                    Y_cord = self.car.Y_cord_cur
                    X_velo = self.car.X_velo
                    Y_velo = self.car.Y_velo
                    
                    # retrieve the q-values of the next state; get argmax
                    q_vals_prime = self.q_table[X_cord, Y_cord, X_velo, Y_velo]
                    q_prime_max = np.max(q_vals_prime)
                    
                    # compute the new q-value and update the q-table
                    disc_reward = self.r_discount * q_prime_max
                    q_val_update = (self.car.env.reward + disc_reward - q_val)
                    q_val_update *= self.r_learning
                    q_vals[action_idx] = q_val_update
                
                # innner loop stopping criterion
                ep_itr += 1
                if ep_itr == self.max_itr: done = True
            
            # decay: gradually decrease the exploration probability 
            # and the learning rate during each iteration
            
            self.p_explore *= self.r_decay
            
            if self.r_learning > 0.01:
                self.r_learning *= self.r_decay
            
            # finally: update the running performance table
            self.training_results[episode] = ep_itr

            
    def test(self):
        '''
        testing simulator for the algorithm. executes the learned policy from
        training on a fresh raceterack environment. 
        '''
        self.car.restart_env() # reset the car's state; place at starting line
        
        # iterate until either the agent has reached the finish 
        # line or the 'max_itr' is hit
        
        test_itr = 0
        done = False
        
        while not done:
            
            # retrive the current state of the car
            X_cord = self.car.X_cord_cur
            Y_cord = self.car.Y_cord_cur
            X_velo = self.car.X_velo
            Y_velo = self.car.Y_velo
            
            s = (X_cord, Y_cord, X_velo, Y_velo)
            
            # retieve the q-values for the current state
            q_vals = self.q_table[X_cord,Y_cord, X_velo, Y_velo]
            
            # action selection: apply explore vs. exploit strategy
            action, action_idx, q_val = epsilon_greedy(self.car, q_vals, self.p_explore)
            
            self.car.update_state(action) # perform the action
            
            # retrive the next state of the car
            X_cord = self.car.X_cord_cur
            Y_cord = self.car.Y_cord_cur
            X_velo = self.car.X_velo
            Y_velo = self.car.Y_velo
            
            s_prime = (X_cord, Y_cord, X_velo, Y_velo)
            
            print(s, action, s_prime)
            
            # stopping criteria: check if car has finished or if the 
            # max_itr has been hit; if true, terminate
            test_itr += 1
            
            if self.car.is_finished: done = True
            if test_itr >= 500: done = True
            
