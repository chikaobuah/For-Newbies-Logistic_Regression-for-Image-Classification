import numpy as np
"""
    Compute the sigmoid of z

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
def sigmoid(z):
   
    s = 1 / (1 + np.exp(-z))
     
    return s