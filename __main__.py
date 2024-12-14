# -*- coding: utf-8 -*-
"""
contains an overhead 'Experiment' class used to test the Racetrack problem
with the RL algorithms implemented in the project and collect results

@name:          __main__.py
@author:        J. Tyler Leake
@last update:   08-19-2024
"""

import numpy as np
from statistics import mean
import matplotlib as plt
from Racetrack import *
from Car import *
from ValueIteration import *
from QLearning import *
from SARSA import *



class Experiment:
    
    def __init__(self, 
                 racetrack_path, 
                 crash_type = ['nearest', 'restart'], 
                 algorithm = ['VI', 'QL', 'SARSA'], 
                 n_experiments = 10, 
                 n_rand_samples = 100):
                 
        # required attrributes
        self.env = Racetrack(racetrack_path)
        self.car = Car(self.env, crash_type)
        self.alg = algorithm
        
        # results 
        self.n_experiments = n_experiments 
        self.cumulative_rewards = None
        self.learning_curve_data = None
        
        # hyperparameter attributes
        self.n_rand_samples = n_rand_samples
        self.best_hyparams = {}
        
        self.candidate_hyperparams = {
        'learning rate'     : np.linspace(0.01, 0.5, 25),
        'discount rate'     : np.linspace(0.95, 0.99, 5),
        'decay rate'        : np.linspace(0.95, 0.99, 5),
        'epsilon'           : np.linspace(0.01, 1, 100),
        'theta'             : np.linspace(0.01, 0.1, 10)}

    def run_procedure(self):
        '''
        overhead procedure for experiment. find the best hyperparameter 
        values using random search, then test with best hyperparameters
        '''
        self.random_search()
        self.train_and_test(self.alg, self.best_hyparams, tuning = False)
    
    def random_search(self):
        '''
        implementation of random search for hyperparameter tuning. trains
        and tests a model on each hyperparam set, recording results
        '''
        # get random hyperparamter samples
        hyparams = self.get_rand_samples() 
        
        # for each hyperparameter sample, train and test a model using
        # the relevant algorithm. record the results
        hyparam_results = {}
        for hyparam_set in hyparams:
            results = self.train_and_test(self.alg, hyparam_set, tuning = True)
            mean_result = mean(list(results.values()))
            hyparam_results[tuple(hyparam_set.items())] = mean_result
        
        # identify and store the best set of hyperparameters
        self.best_hyparams = dict(min(hyparam_results, key=hyparam_results.get))

    def get_rand_samples(self):
        '''
        generates random hyperparameter sample for random search tuning
        '''
        param_sets = []
        for _ in range(self.n_rand_samples):
            sampled_params = {}
            for hyp_param, param_range in self.candidate_hyperparams.items():
                if isinstance(param_range, list):
                    sample = np.random.choice(param_range)
                else:
                    sample = np.random.uniform(min(param_range), max(param_range))
                sampled_params[hyp_param] = sample
            param_sets.append(sampled_params)
        return param_sets
    
    def train_and_test(self, algorithm, hyparams, tuning = False):
        '''
        trains a model using the experiment's attribute using the racetrack 
        env attribute; returns the experiment results 
        '''
        
        r_learning = hyparams['learning rate']
        r_discount = hyparams['discount rate']
        r_decay = hyparams['decay rate']
        p_explore = hyparams['epsilon']
        
        train_performance = {}
        test_performance = {}
        Lcurve_data = {}
        
        for exp_no in range(self.n_experiments):
        
            if algorithm == 'VI':
                exp = ValueIteration(self.car, r_learning, r_discount, r_decay, p_explore)
                exp.train()
                exp.test()
                train_performance[exp_no] = exp.training_results
                test_performance[exp_no] = exp.test_results
                if not tuning: Lcurve_data[exp_no] = list(exp.training_results.values())
            
            if algorithm == 'QL':
                exp = QLearning(self.car, r_learning, r_discount, r_decay, p_explore)
                exp.train()
                exp.test()
                mean_train_steps = mean(list(exp.training_results.values()))
                mean_test_steps = mean(list(exp.testimg_results.values()))
                train_performance[exp_no] = mean_train_steps
                test_performance[exp_no] = mean_test_steps
                test_performance[exp_no] = mean_test_steps
                if not tuning: Lcurve_data[exp_no] = list(exp.training_results.values())
                
            if algorithm == 'SARSA':
                exp = SARSA(self.car, r_learning, r_discount, r_decay, p_explore)
                exp.train()
                exp.test()
                mean_train_steps = mean(list(exp.training_results.values()))
                mean_test_steps = mean(list(exp.testimg_results.values()))
                train_performance[exp_no] = mean_train_steps
                test_performance[exp_no] = mean_test_steps
                test_performance[exp_no] = mean_test_steps
                if not tuning: Lcurve_data[exp_no] = list(exp.training_results.values())
                    
        if tuning:
            return experiments
        
        if not tuning:
            self.cumulative_rewards = cumulative_rewards
            self.learning_curve_data = learning_curve_data
            
        