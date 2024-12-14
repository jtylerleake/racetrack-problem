# -*- coding: utf-8 -*-
"""
contains the 'Racetrack' class representing the environment of the 
racetrack problem. 

@name:          Racetrack.py
@author:        J. Tyler Leake
@last update:   08-19-2024
"""

import numpy as np
import random
from itertools import combinations
from math import sqrt



class Racetrack(): 
    
    def __init__(self, 
                 env_path, 
                 p_transition= 0.80, 
                 reward = -1, 
                 accl_range = (-1, 1), 
                 X_velo_range = (-5, 5), 
                 Y_velo_range = (-5, 5)):
        
        # environment dimensions (x,y coordinates, velocities)
        self.X_cord_dim = None
        self.Y_cord_dim = None
        self.X_velo_dim = (X_velo_range[0], X_velo_range[1])
        self.Y_velo_dim = (Y_velo_range[0], Y_velo_range[1])
        
        # map representation of the track and coordinates for the 
        # start, finish, wall and track spaces
        self.map_rep = self.load_env(env_path)
        self.start_cords = self.get_coordinates('S')
        self.finish_cords = self.get_coordinates('F')
        self.track_cords = self.get_coordinates('.')
        self.wall_cords = self.get_coordinates('#')
        
        # velocity and acceleration attr.
        self.actions = self.get_actions(accl_range)
        
        # reward and transition function attributes
        self.reward = reward
        self.p_transition = p_transition
        
        # finish line boundary and orientation helpers
        self.fbound1 = None
        self.fbound2 = None
        self.is_vert_finish = None
        
        self.get_finish_orientation()
        
    def load_env(self, env_path):
        '''
        loads the dataset in from the filepath and converts to np
        array; stores the x,y coordinate dimensions from file line 1
        '''
        with open(env_path) as env_data: lines = env_data.readlines()
        lines = [line.strip() for line in lines]
        env_dims = lines.pop(0)
        env_dims = env_dims.split(',')
        self.X_cord_dim, self.Y_cord_dim = int(env_dims[0]), int(env_dims[1])
        env_arr = [list(line) for line in lines]
        return np.array(env_arr)

    def get_coordinates(self, char):
        '''
        returns the coordinates in the array where the 'char' arg 
        is found; used to find start/finish coordinates
        '''
        coordinates = np.where(self.map_rep == char)
        X_cords, Y_cords = coordinates[0], coordinates[1]
        coordinates = [(X_cords[i], Y_cords[i]) for i in range(len(X_cords))]
        return coordinates
    
    def get_actions(self, accl_range):
        '''
        returns the set of possbile actions for the agent in the 
        environment from the acceleration range provided as a list 
        '''
        accl_range = np.arange(accl_range[0], accl_range[1] + 1)
        actions = [(X_accl, Y_accl) for Y_accl in accl_range for X_accl in accl_range]
        return actions
    
    def get_rand_start(self):
        '''
        selects one of the starting points in the env. randomly
        '''
        return random.choice(self.start_cords)
    
    def get_finish_orientation(self):
        '''
        determines whether the finish line is vertically or 
        horizontally oriented
        '''
        
        # compute distance between each 2-combination of coordinates 
        # in the finish line and determine farthest pair
        max_distance = 0
        farthest_pairs = None
        
        for fin1, fin2 in combinations(self.finish_cords, 2):

            distance = sqrt((fin1[0] - fin2[0])**2 + (fin1[1] - fin2[1])**2)
            
            if distance > max_distance: 
                farthest_pairs = (fin1, fin2)
                max_distance = distance
        
        self.fbound1, self.fbound2 = farthest_pairs[0], farthest_pairs[1]
        
        # if the farthest coordinates have same x-axis value, the line 
        # is horizontal, else it is vertical
        self.is_vert_finish = True
        
        if farthest_pairs[0][0] == farthest_pairs[1][0]: 
            self.is_vert_finish  = False   

