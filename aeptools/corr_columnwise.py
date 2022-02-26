def corr_columnwise(X1, X2):
    '''
    Purpose : compute column-wise correlation coefficients
    
    Introduction
    ------------
    - Matlab's corr() computes the correlation of columns between X1 and X2.
    - Numpy's corrcoef() calculates the correlation of rows within an array.
    - Both input arrays will be normalized before calculating correlation coefficients
   
    Parameters
    ----------
    X1 : numpy array (2D)
        - input 1
    X2 : numpy array (2D)
        - input 2
        
    Returns
    -------
    coefficients : tuple of float64
        - column-wise correlation coefficients
        
    Diary
    ---------------------------
    2022-02-13 (v01) 
        - Fuh wrote this script in Python
        
    ToDo
    ----
    # TODO: check input matrix/vector size
    # TODO: if the input matrix/vector size is inappropriate, correct them. 
                
    '''
    import numpy as np
    
    normalizeFlag = True    # whether to normalize input vectors manually before doing cross-correlation, or not
    if normalizeFlag:
        X1 = (X1 - X1.mean(axis=0)) / X1.std(axis=0)           # normalize X1, prior to corrcoef()
        X2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)           # normalize X2, prior to corrcoef()
    coefficients = (np.dot(X2.T, X1) / X2.shape[0])[0]
       
    return coefficients
