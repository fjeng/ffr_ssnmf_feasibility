def corr_columnwise(X1, X2, normalizeFlag=True):
    '''
    Purpose : compute column-wise correlation coefficients
    
    Introduction
    ------------
    - Matlab's corr() computes the correlation of columns between X1 and X2.
    - Numpy's corrcoef() calculates the correlation of rows within an array.
    - This script mimics the funcationality of Matlab's corr() by using a custom-written script in Python.
   
    Parameters
    ----------
    X1 : numpy array (2D)
        - input 1
    X2 : numpy array (2D)
        - input 2
    normalizeFlag : boolean
        - whether to normalize input vectors before computing correlation coefficients, or not [default = True]
        
    Returns
    -------
    coefficients : tuple of float64
        - column-wise correlation coefficients
        
    Diary
    ---------------------------
    2022-02-13 (v01) 
        - Fuh-Cherng (Fuh) Jeng wrote this script in Python
        
    ToDo
    ----
    # TODO: check input matrix/vector size
    # TODO: if the input matrix/vector size is inappropriate, show an error message. 
                
    '''
    import numpy as np
    
    normalizeFlag = True   
    if normalizeFlag:
        X1 = (X1 - X1.mean(axis=0)) / X1.std(axis=0)           
        X2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)           
        
    coefficients = (np.dot(X2.T, X1) / X2.shape[0])[0]
       
    return coefficients
