# import numpy as np

# # Write a function that takes as input two lists Y, P,
# # and returns the float corresponding to their cross-entropy.
# def cross_entropy(Y, P):
#     D = []
#     for y,p in zip(Y,P):
#         if y == 0:
#           D.append(-np.log(1-p))
#         else:
#           D.append(-np.log(p))
    
#     return sum(D)

import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    # D = []
    # for y,p in zip(Y,P):
    #     if y == 0:
    #       D.append(-np.log(1-p))
    #     else:
    #       D.append(-np.log(p))
    
    # return sum(D)
    Y = np.float_(Y)
    P = np.float_(P)
    return -sum((Y*np.log(P)) + ((1-Y)*(np.log(1-P))))