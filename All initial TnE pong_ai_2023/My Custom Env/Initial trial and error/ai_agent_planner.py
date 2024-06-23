from math_functions import log_stable
from math_functions import softmax
from math_functions import kl_div
import numpy as np
import math
from pymdp.maths import spm_log_single as log_stable

EPS_VAL = 1e-16 #negligibleconstant

def entropy(A):
    """ Compute the entropy of a set of condition distributions, i.e. one entropy value per column """
    
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A

#Active inference algorithm
def action_dist(A,B,C,T,sm_par):
    
    x = B[0].shape
    xx = C.shape
    
    numS = x[0]
    numA = x[2]
    numO1 = C.shape[0]

    Bo1_i = np.zeros((numO1,numS,numA))

    for i in range(numS):
        for j in range(numA):
            Bo1_i[:,i,j] = A.dot(B[0][:,i,j])


    Q_po = Bo1_i 
    C_po = C

    #planning horizon
    G=np.zeros((T-1,numA,numS))
    Qpi=np.zeros((T-1,numA,numS))
    
    # entropy(A).dot(B[0])[j,i]
    for k in range(T-2,-1,-1):
        for i in range(numA):
            for j in range(numS):

                if(k==T-2):
                    G[k,i,j]=kl_div(np.array(Q_po[:,j,i]).flatten(),np.array(C_po).flatten()) #+ entropy(A).dot(B[0])[j,i]
 
                else:
                    G[k,i,j]=kl_div(np.array(Q_po[:,j,i]).flatten(),np.array(C_po).flatten()) #+ entropy(A).dot(B[0])[j,i]
                    for jj in range(numS):
                        for kk in range(numA):
                            G[k,i,j]+=Qpi[k+1,kk,jj]*B[0][jj,j,i]*G[k+1,kk,jj]

        #Distribution for action-selection
        for ppp in range(numS):
            Qpi[k,:,ppp]=softmax(sm_par*(-1*G[k,:,ppp]))
    
    return(Qpi)