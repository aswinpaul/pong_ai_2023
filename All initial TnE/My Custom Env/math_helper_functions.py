EPS_VAL = 1e-16 #negligibleconstant

import numpy as np
import math
from scipy.stats import dirichlet

def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """

    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def log_stable(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(arr + EPS_VAL)

def kl_div(P,Q):
    n=len(P)
    for i in range(n):
        if(P[i]==0):
            P[i]+=EPS_VAL
        if(Q[i]==0):
            Q[i]+=EPS_VAL
            
    dkl=0
    for i in range(n):
        dkl+=(P[i]*math.log(P[i]))-(P[i]*math.log(Q[i]))
    return(dkl)

def obj_array(num_arr):
    """
    Creates a generic object array with the desired number of sub-arrays, given by `num_arr`
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


def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr

# Normalising A and B as probability distributions

def normalise_A(A, num_states, num_modalities):
    for j in range(num_modalities):
        A[j] = A[j] / A[j].sum(axis=0)[np.newaxis,:]
    return A

def normalise_B(B, num_states, num_controls):
    for i in range(len(num_states)):            
        for j in range(num_states[i]):
            for k in range(num_controls[i]):
                B[i][:,j,k]=dirichlet.mean(B[i][:,j,k])
            
    return B

def pong_state_to_obs(state, reward, factor):
    
    ball_x = state[0]
    o1_float = ball_x/factor
    o1 = int(o1_float)

    ball_y = state[1] 
    o2_float = ball_y/factor
    o2 = int(o2_float)

    ball_vx = state[2]
    o3_float = ball_vx
    o3_i = int(o3_float)
    o3 = 0 if o3_i==2 else 1 


    ball_vy = state[3]
    o4_float = ball_vx
    o4_i = int(o4_float)
    o4 = 0 if o4_i==2 else 1 

    paddle_pos = state[4]
    o5_float = paddle_pos/factor
    o5 = int(o5_float)

    paddle_vel = state[5]
    o6_float = paddle_vel
    o6_i = int(o6_float)
    o6 = 0 if o6_i==4 else 1 
    
    if(reward == -1):
        o7 = 0
    elif(reward == 1):
        o7 = 2
    else:
        o7 = 1
    
    observation = [o1, o2, o3, o4, o5, o6, o7]
    
    return(observation)