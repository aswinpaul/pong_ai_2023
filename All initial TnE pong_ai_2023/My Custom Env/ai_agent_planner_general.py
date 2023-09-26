from math_helper_functions import log_stable
from math_helper_functions import softmax
from math_helper_functions import kl_div
import numpy as np
import math
import random
import pymdp
from pymdp.maths import spm_log_single as log_stable
from pymdp import utils

EPS_VAL = 1e-16 #negligibleconstant

random.seed(123)
np.random.seed(123)

def entropy(A):
    """ Compute the entropy of a set of condition distributions, i.e. one entropy value per column """
    
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A

#Dynamic programming in G (expected free energy)

def action_dist(A, B, C, T, sm_par):
    
    num_modalities = A.shape[0]
    num_factors = B.shape[0]

    num_states = []
    for i in range(num_factors):
        num_states.append(B[i].shape[0])

    num_obs = []
    for i in range(num_modalities):
        num_obs.append(A[i].shape[0])

    num_controls = []
    for i in range(num_factors):
        num_controls.append(B[i].shape[2])

    numS = 1
    for i in num_states:
        numS *= i
    numA = 1
    for i in num_controls:
        numA *= i

    new_num_states = [numS]
    new_num_controls = [numA]

    new_A = utils.random_A_matrix(num_obs, new_num_states) #* 0 + EPS_VAL
    new_B = utils.random_B_matrix(1, 1) #* 0 + EPS_VAL

    for i in range(num_modalities):
        new_A[i] = np.reshape(A[i], [A[i].shape[0], numS])

    for i in range(num_factors):
        new_B[0] = np.kron(new_B[0],B[i])

    #Expected free energy (Only RISK)
    
    G = np.zeros((T-1, numA, numS))
    Q_actions = np.zeros((T-1, numA, numS))

    for mod in range(6,num_modalities): #Only the last modality for planning

        Q_po = np.zeros((A[mod].shape[0], numS, numA))

        for i in range(numS):
            for j in range(numA):
                Q_po[:,i,j] = new_A[mod].dot(new_B[0][:,i,j])

        for k in range(T-2,-1,-1):
            for i in range(numA):
                for j in range(numS):

                    if(k==T-2):
                        G[k,i,j] += kl_div(Q_po[:,j,i],C[mod]) + np.dot(new_B[0][:,j,i],entropy(new_A[mod]))

                    else:
                        G[k,i,j] += kl_div(Q_po[:,j,i],C[mod]) + np.dot(new_B[0][:,j,i],entropy(new_A[mod]))
                        for jj in range(numS):
                            for kk in range(numA):
                                G[k,i,j] += Q_actions[k+1,kk,jj]*new_B[0][jj,j,i]*G[k+1,kk,jj] 

            #Distribution for action-selection
            for l in range(numS):
                Q_actions[k,:,l] = softmax(sm_par*(-1*G[k,:,l]))
                
    return Q_actions