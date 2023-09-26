#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:42:22 2022

@author: aswinpaul
"""
import numpy as np
from scipy.stats import dirichlet
np.random.seed(2022)

# Helper functions from pymdp: https://github.com/infer-actively/pymdp

def log_stable(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    EPS_VAL = 1e-10
    return np.log(arr + EPS_VAL)

def kl_div(P,Q):
    """
    Parameters
    ----------
    P : Categorical probability distribution
    Q : Categorical probability distribution

    Returns
    -------
    The KL-DIV of P and Q

    """          
    dkl = 0 
    for i in range(len(P)):
        dkl += (P[i]*log_stable(P[i]))-(P[i]*log_stable(Q[i]))
    return(dkl)

def entropy(A):
    """ 
    Compute the entropy of a set of condition distributions, 
    i.e. one entropy value per column 
    """
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A

def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr

def obj_array(num_arr):
    """
    Creates a generic object array with the desired number of sub-arrays, 
    given by `num_arr`
    """
    return np.empty(num_arr, dtype=object)

def obj_array_zeros(shape_list):
    """ 
    Creates a numpy object array whose sub-arrays are 1-D vectors
    filled with zeros, with shapes given by shape_list[i]
    """
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = np.zeros(shape)
    return arr

def norm_dist(dist):
    """ Normalizes a Categorical probability distribution (or set of them) 
    assuming sufficient statistics are stored in leading dimension"""
    
    if dist.ndim == 3:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))

def random_A_matrix(num_obs, num_states):
    """ Generates a random A-matrix i.e liklihood matrix using number of state and observation modalitiles
    """
    if type(num_obs) is int:
        num_obs = [num_obs]
    if type(num_states) is int:
        num_states = [num_states]
    num_modalities = len(num_obs)

    A = obj_array(num_modalities)
    for modality, modality_obs in enumerate(num_obs):
        modality_shape = [modality_obs] + num_states
        modality_dist = np.random.rand(*modality_shape)
        A[modality] = norm_dist(modality_dist)
    return A

def random_B_matrix(num_states, num_controls):
    """Generates a random B matrix i.e one step dynamics matrix using the number of (hidden states) and number of controls in each hidden states.
    Minimum number of controls equal to one i.e markov chain with action: 'Do nothing'.
    """
    if type(num_states) is int:
        num_states = [num_states]
    if type(num_controls) is int:
        num_controls = [num_controls]
    num_factors = len(num_states)
    assert len(num_controls) == len(num_states)

    B = obj_array(num_factors)
    for factor in range(num_factors):
        factor_shape = (num_states[factor], num_states[factor], num_controls[factor])
        factor_dist = np.random.rand(*factor_shape)
        B[factor] = norm_dist(factor_dist)
    return B


def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """

    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def normalise_C(A, num_factors):
    for j in range(num_factors):
        A[j] = A[j] / A[j].sum(axis=0)[np.newaxis,:]
    return A

# CL Method AGENT CLASS (Author: Aswin Paul)

class agent():
    
    def __init__(self, num_states, num_obs, num_controls, horizon = 5, 
                 gamma_initial = 0.55, a = 0, b = 0, d = 0, MDP=True):
        # Generative model

        self.numS = 1
        self.numA = 1
        for i in num_states:
            self.numS *= i
        for i in num_controls:
            self.numA *= i
            
        self.num_states = [self.numS]
        self.num_factors = len(self.num_states)
        self.num_controls = [self.numA]
        self.num_actions = self.numA
        self.num_obs = num_obs
        self.num_modalities = len(num_obs)
        
        self.EPS_VAL = 1e-16
        self.a = random_A_matrix(self.num_obs, self.num_states)*0 + self.EPS_VAL
        
        if(type(a) != int):
            for i in range(len(self.num_obs)):
                self.a[i] = a[i].reshape(num_obs[i], self.numS) 
        
        self.b = random_B_matrix(self.num_states, self.num_controls)*0 + self.EPS_VAL
        if(type(b) != int):
            bb = 1
            for i in range(len(num_states)):
                bb = np.kron(bb, b[i])
            self.b[0] = bb
            
        if(type(d) != int):
            self.d = d
        else:
            self.d = obj_array_zeros(self.num_states)
            for idx in range(len(self.num_states)):
                self.d[idx] += 1 / self.num_states[idx]

        self.A = random_A_matrix(self.num_obs, self.num_states)*0 + self.EPS_VAL
        self.B = random_B_matrix(self.num_states, self.num_controls)*0 + self.EPS_VAL
        self.D = obj_array_zeros(self.num_states)
        
        self.qs = obj_array_zeros(self.num_states)
        self.qs = np.copy(self.D)
        
        self.qs_prev = obj_array_zeros(self.num_states)
        self.qs_prev = np.copy(self.qs)
        self.horizon = horizon
        self.MDP = MDP
        
        self.a += self.EPS_VAL
        self.learn_parameters()
        ## State to decision mapping

        # Policy state mapping C
        
        self.C = obj_array_zeros([1])
        self.C[0] = np.random.rand(self.numA, self.numS)*0 + self.EPS_VAL
        self.C = normalise_C(self.C, self.num_factors)
        
        # Risk (Gamma)
        self.Gamma = np.zeros((horizon, 1))
        for i in range(self.horizon):
            self.Gamma[i,0] = gamma_initial
            
        self.last_n_q = []
        self.last_n_action = []
    
    # Inference using belief propogation (BP)
    def infer_hiddenstate(self, obs):
        self.qs_prev = np.copy(self.qs)
        
        for i in range(len(self.num_states)):
            # Likelihood
            term_2 = 0
            for j in range(len(self.num_obs)):
                term_2 += log_stable(np.matmul(np.transpose(self.A[j]),
                                               onehot(obs[j],self.num_obs[j])))
            
            if(self.MDP == True):
                # Only likelihood
                self.qs[i] = softmax(term_2)
            else:
                # Prior when POMDP
                if(self.tau == 0):
                    term_1 = log_stable(self.D[i] + self.EPS_VAL)
                else:
                    term_1 = log_stable(np.matmul(self.B[i][:,:,self.action],
                                                  self.qs_prev[i]))
                
                #Equal-weightage for prior and likelihood
                self.qs[i] = softmax(term_1 + term_2)
        
        if(len(self.last_n_q) > self.horizon):
            self.last_n_q.pop(0)
        self.last_n_q.append(self.qs_prev)
                
    def take_decision(self):
            c_s = np.matmul(np.log(self.C[0]), self.qs[0])
            p_d = softmax(c_s)
            action = np.random.choice(list(range(self.numA)), size=None, replace=True, p=p_d)
            self.action = action
            
            if(len(self.last_n_action) > self.horizon):
                self.last_n_action.pop(0)
            self.last_n_action.append(self.action)
            
            return action
            
    def update_gamma(self,hit=False,miss=False):
        if(hit):
            for ii in range(self.horizon):
                self.Gamma[ii] -= 1 #/(self.horizon - ii)
        elif(miss):
            for ii in range(self.horizon):
                self.Gamma[ii] += 1 #/(self.horizon - ii)
        self.Gamma = np.clip(self.Gamma, 0, 1)
        #print(self.Gamma)
        
    def update_C(self, tau):
        if(tau < self.horizon):
            t = tau
        else:
            t = self.horizon
            
            count = self.horizon
            for i in range(t):
                count -= 1
                
                qs_prev = self.last_n_q[i]
                action = self.last_n_action[i]
                
                # Learning C
                des = onehot(action,self.num_actions)
                a = np.reshape(des, (des.shape[0],1))
                b = np.reshape(qs_prev[0], (1,qs_prev[0].shape[0]))
                c_up = np.kron(a,b)
                
                self.C[0] += (1 - 2*self.Gamma[i][0])*c_up
                self.C[0] = np.clip(self.C[0], self.EPS_VAL, None)
                self.C = normalise_C(self.C, self.num_factors)
            
    # Learning parameters A,B,C,D using a,b,c,d
    def learn_parameters(self, factor=1):
        for i in range(self.num_modalities):
            for k in range(self.num_states[0]):
                self.A[i][:,k] = dirichlet.mean(factor*self.a[i][:,k])
        
        for i in range(len(self.num_states)):
            for j in range(self.num_states[i]):
                for k in range(self.num_controls[i]):
                    self.B[i][:,j,k] = dirichlet.mean(factor*self.b[i][:,j,k])
                
        for i in range(len(self.num_states)):
            self.D[i] = softmax(self.d[i])