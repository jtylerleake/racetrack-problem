# -*- coding: utf-8 -*-
"""
contains the 'Car' class, representing the agent in the racetrack problem. 
the racetrack environment is stored as an attribute for the car 

@name:          Car.py
@author:        J. Tyler Leake
@last update:   08-19-2024
"""

import numpy as np



class Car:
    
    def __init__(self, Racetrack, crash_type = ['nearest', 'restart']):
        
        # agent's enviornment + attributes for crash/finish
        self.env = Racetrack 
        self.crash_type = crash_type
        self.is_finished = False
        
        # agent's state values in the environment
        self.X_velo = 0
        self.Y_velo = 0
        self.X_cord_cur = None
        self.Y_cord_cur = None
        self.X_cord_old = None
        self.Y_cord_old = None
        
        self.restart_env() # initialize car at starting line

    def restart_env(self):
        '''
        places the car at one of the starting points in the map
        and sets the velocity to zero
        '''
        # reset the x/y velocities to zero and the old x/y coordinates 
        # to none; set the is_finished attr. to False
        self.X_velo = 0
        self.Y_velo = 0
        self.X_cord_old = None
        self.Y_cord_old = None
        self.is_finished = False
        
        # retrieve a random starting cordinate; set as current x/y coordinates
        restart_cord = self.env.get_rand_start()
        self.X_cord_cur, self.Y_cord_cur = restart_cord[0], restart_cord[1]
        
    def update_state(self, action):
        '''
        updates the coordinate and velocity state values of the 
        car based on the action
        
        args: 
        action (np arr): x/y acceleration action to apply to agent
            
        return:
        none; agent's state values updated directly
        '''
        
        X_accl, Y_accl = action[0], action[1]
        
        # store old x/y coordinates
        self.X_cord_old, self.Y_cord_old = self.X_cord_cur, self.Y_cord_cur
        
        # compute the new velocity; assert it does not exceed speed limits
        X_velo_new = self.X_velo + X_accl
        Y_velo_new = self.Y_velo + Y_accl
        self.X_velo = max(self.env.X_velo_dim[0],min(X_velo_new, self.env.X_velo_dim[1]))
        self.Y_velo = max(self.env.Y_velo_dim[0],min(Y_velo_new, self.env.Y_velo_dim[1]))
        
        # update the car's coordinates state values
        self.X_cord_cur += X_velo_new
        self.Y_cord_cur += Y_velo_new
        
        self.check_if_finished() # check if car has reached finish line
        self.crash_procedure() # run crash procedure
        
    def crash_procedure(self):
        '''
        determines if the agent has crashed in its environment. if True, 
        place the car at either the nearest position or starting line
        '''
        cur_cords = (self.X_cord_cur, self.Y_cord_cur)
        
        # check if the car's coordinates are in a wall
        in_wall = True if cur_cords in self.env.wall_cords else False
        
        # check if the car's coordinates are off of the map
        exceeded_bounds = False
        
        if (self.X_cord_cur + 1 > self.env.X_cord_dim or self.X_cord_cur < 0) or \
           (self.Y_cord_cur + 1 > self.env.Y_cord_dim or self.Y_cord_cur < 0):
               
            exceeded_bounds = True
    
        # if in a wall or off the map, place car according to crash policy
        if in_wall or exceeded_bounds:
            
            # place at nearest coordinate in the track env.
            if self.crash_type == 'nearest':
                
                cur_cords = np.array(cur_cords)
                relief_cords = np.array(self.env.track_cords+self.env.start_cords)
                relief_distances = np.sum(np.square(relief_cords-cur_cords), axis = 1)
                    
                min_relief_idx = np.argmin(relief_distances)
                min_relief = relief_cords[min_relief_idx]
                
                # place the car the nearest track coordinate; reset speed
                self.X_cord_old, self.Y_cord_old = None, None
                self.X_cord_cur, self.Y_cord_cur = min_relief[0], min_relief[1] 
                self.X_velo, self.Y_velo = 0, 0
           
            # place the car at the starting line; reset speed
            if self.crash_type == 'restart':
                self.restart_env()
            
    def check_if_finished(self):
        '''
        determines if the agent's current coordinates are in the finish 
        line coordinates of the environment; updates 'is_finished' attr.
        '''
        crossed = False
        within_fbounds = False
        cur_cords = (self.X_cord_cur, self.Y_cord_cur)
        old_cords = (self.X_cord_old, self.Y_cord_old)
        
        # vertical finish line 
        if self.env.is_vert_finish:
            
            # compute the agent's distance to the finish line 
            # before and after the action
            dist_pre_act = (self.env.fbound1[1] - old_cords[1])
            dist_pst_act = (self.env.fbound1[1] - cur_cords[1])
            
            # vertical test: if product of distances is negative, 
            # then agent did pass vertically over the finish line
            if (dist_pre_act * dist_pst_act) <= 0: 
                
                crossed = True 
            
                # horizontal test: if agent was within finish bounds before 
                # or after crossing, then it crossed within the bounds
                if (self.env.fbound1[0] <= old_cords[0] <= self.env.fbound2[0]) or \
                   (self.env.fbound1[0] <= cur_cords[0] <= self.env.fbound2[0]):
                       
                       # within_fbounds = True
                       # if crossed and within_bounds
                       self.is_finished = True
            
        # horizontal finish line    
        if not self.env.is_vert_finish: 
            
            # compute the agent's distance to the finish line 
            # before and after the action
            dist_pre_act = (self.env.fbound1[0] - old_cords[0])
            dist_pst_act = (self.env.fbound1[0] - cur_cords[0])
            
            # horizontal test: if product of distances is negative, 
            # then agent did pass horizontally over the finish line
            if (dist_pre_act * dist_pst_act) <= 0: 
                
                crossed = True 
            
                # vertical test: if agent was within finish bounds before 
                # or after crossing, then it crossed within the bounds
                if (self.env.fbound1[1] <= old_cords[1] <= self.env.fbound2[1]) or \
                   (self.env.fbound1[1] <= cur_cords[1] <= self.env.fbound2[1]):
                       
                       # within_fbounds = True
                       # if crossed and within_bounds
                       self.is_finished = True
                       