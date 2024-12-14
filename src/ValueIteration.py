# -*- coding: utf-8 -*-
"""
contains implementation of the value iteration algorithm for the 
racetrack problem as a class. 

@name:          ValueIteration.py
@author:        J. Tyler Leake
@last update:   08-19-2024
"""

import numpy as np
from Car import *
from utils import init_q_table



class ValueIteration:

    def __init__(self, car, theta, r_discount, max_itr = 100):
        
        # racetrack environment to train on
        self.car = car
        self.env = car.env
        
        # state value table; action-value table; and policy table
        self.v_table = None
        self.q_table = None
        self.p_table = None

        # model stopping criteria: either convergence 
        # threshold or max number of iterations tolderated
        self.theta = theta
        self.max_itr = max_itr
        self.r_discount = r_discount
        
        # results
        self.training_results = 0
        
    def init_v_table(self):
        '''
        initializes the state value table for the algorithm with all 
        possible states set to zero
        '''
        X_cord_dim = self.env.X_cord_dim
        Y_cord_dim = self.env.Y_cord_dim
        X_velo_dim = abs(self.env.X_velo_dim[1] - self.env.X_velo_dim[0])
        Y_velo_dim = abs(self.env.Y_velo_dim[1] - self.env.Y_velo_dim[0])
        return np.zeros([X_cord_dim, Y_cord_dim, X_velo_dim, Y_velo_dim])
        
    def train(self):
        '''
        implementation of the value iteration algorithm 
        '''
        car = self.car

        # initialize the state value table V(s); the action-value 
        # table Q(s); and policy table P
        self.v_table = self.init_v_table()
        self.q_table = init_q_table(self.env)
        self.p_table = self.init_v_table()

        # stopping criteria: train until either the delta val has reached 
        # the threshold or the max number of iterations has been reached
        
        done = False
        
        while not done: 

            max_q_delta = 0 
        
            # for each state in the environment
            for X_cord in range(self.env.X_cord_dim):
                for Y_cord in range(self.env.Y_cord_dim):
                    for X_velo in range(10):
                        for Y_velo in range(10):
                            
                            # retrieve the current state value from the table
                            old_state_val = self.v_table[X_cord, Y_cord, X_velo, Y_velo]
            
                            # for each action the car can taken in the env,
                            # find the state/action pair with max reward
                            
                            max_action = None
                            max_q_val = -10000
                            
                            for action in self.env.actions:
                                
                                # retrieve the car's coordinates and velocity attr.
                                car.cur_cord = [X_cord, Y_cord]
                                car.cur_velo = [X_velo, Y_velo]
                                
                                car.update_state(action) # perform the action
                                
                                # store the new state values from the action
                                X_cord_new = car.cur_cord[0]
                                Y_cord_new = car.cur_cord[1]
                                X_velo_new = car.cur_velo[0]
                                Y_velo_new = car.cur_velo[1]
                                
                                # check if the car has finished. if true, set the 
                                # reward to 0; else retrieve next state from v_table
                                
                                next_state_val = None
                                reward = self.env.reward
                                
                                if car.is_finished: 
                                    reward = 0 
                                
                                if not car.is_finished: 
                                    next_state_val = self.v_table[X_cord_new, Y_cord_new, 
                                                                  X_velo_new, Y_velo_new]
                                    
                                # compute the q-value of the state/action pair;
                                # insert this value into q_table
                                q_val = reward + (self.r_discount * next_state_val)
                                self.q_table[X_cord, Y_cord, X_velo, Y_velo, action] = q_val
                                
                                # update outer loop vars if q value is new max
                                if q_val > max_q_val:
                                    best_action = action
                                    max_q_val = q_val
                                    
                            # update the state value table from the best action
                            old_q_val = self.v_table[X_cord, Y_cord, X_velo, Y_velo]
                            self.v_table[X_cord, Y_cord, X_velo, Y_velo] = max_q_val
                            self.p_table[X_cord, Y_cord, X_velo, Y_velo] = max_action
                            
                            # update the max_q_delta var if new delta is greater
                            q_val_delta = (old_q_val - max_q_val)
                            
                            if q_val_delta > max_q_delta: 
                                max_q_delta = q_val_delta
                                  
            # stopping criteria
            itr += 1                        
            done = False if (max_q_delta > self.theta and itr < self.max_iter) else True
                           
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
                
                # retieve the action from the policy and perform it
                action = self.p_table[X_cord,Y_cord, X_velo, Y_velo]
                self.car.update_state(action)
                
                # retrive the next state of the car
                X_cord = self.car.X_cord_cur
                Y_cord = self.car.Y_cord_cur
                X_velo = self.car.X_velo
                Y_velo = self.car.Y_velo
                
                s_prime = (X_cord, Y_cord, X_velo, Y_velo)
                
                # stopping criteria: check if car has finished or if the 
                # max_itr has been hit; if true, terminate
                test_itr += 1
                if self.car.is_finished: done = True
                if test_itr >= 500: done = True
                
                print(s, action, s_prime)
                
