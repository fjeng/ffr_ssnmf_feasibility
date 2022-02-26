def nmf_evaluation(spectrogramSignal, spectrogramData, nsweeplabels, reference_sweeps):
    '''
    Purpose : evaluate NMF performance in terms of correlation coefficients and RMSE
    
    Introduction
    ------------
    - corr is the Pearson correlation coefficient between spectrogramSignal and spectrogramData
    - RMS (root-mean-square) error is the RMS value of the differences between spectrogramSignal and spectrogramData
    - NOTE: When estimating Correlation and RMSEnd , computations are based on "straightened/flattened spectrogram data". That is, 2D spectrograms are reshaped into vectors, prior to computing differnces and correlations.

    Parameters
    ----------
    spectrogramSignal : numpy array (3D)
        - spectrogramSignal contains the spectrograms to be processed. It is a 3 dimensional matrix: number of frequencies x number of time frames x number of spectrograms (e.g., 1000 x 201 x 11)
    spectrogramData : numpy array (3D)
        - spectrogramData contains the spectrograms that will be used as the reference point. It is a 3 dimensional matrix: number of frequencies x number of time frames x number of spectrograms (e.g., 1000 x 201 x 11)
    nsweeplabels : numpy array (1D)
        - nsweeplabels are the nSweeps labels that corresponds to all the spectrograms saved in spectrogramData (and should be the same to those in spectrogramSignal as well) 
    reference_sweeps : integer
        - reference_sweeps is the number of sweeps you want to use for the reference in performance evaluation. For example, reference_sweeps = 8000
        
    - For example, when nFrequencies = 1000, nFrames = 201, nSweeps = [100 250 500 1000:1000:8000] and npermutation = 1, the input parameters can be:
        - spectrogramSignal is a 3D matrix [1000 x 201 x 11]
        - spectrogramData is a 3D matrix [1000 x 201 x 11]
        - nsweeplabels is [100 100 100 100 100 250 250 250 250 250 ... 8000 8000 8000 8000 8000]
        - reference_sweeps can be say 8000

   Returns
   -------
    RMSE : dictionary
        - RMSE is the RMS (root-mean-square) error between spectrogramSignal and spectrogramData
        'results' : numpy array (2D)
            - rmse raw data. It has a shape of (npermutation, nsweepCondition).
        'mean' : numpy array (2D) 
            - mean value across npermutation. It has a shape of (1, nsweepCondition).
        'std' : numpy array (2D) 
            - standard deviation across npermutation. It has a shape of (1, nsweepCondition).
        'se' : numpy array (2D) 
            - standard error across npermutation. It has a shape of (1, nsweepCondition).
    CORR : dictionary
        - CORR is the correlation coefficient between spectrogramSignal and spectrogramData
        'results' : numpy array (2D)
            - correlation coefficients. It has a shape of (npermutation, nsweepCondition).
        'mean' : numpy array (2D) 
            - mean value across npermutation. It has a shape of (1, nsweepCondition).
        'std' : numpy array (2D) 
            - standard deviation across npermutation. It has a shape of (1, nsweepCondition).
        'se' : numpy array (2D) 
            - standard error across npermutation. It has a shape of (1, nsweepCondition).
    nsweeplabels_unique : numpy array (1D)
        - an array of unique nsweep labels

    Diary
    -----
    2019-09-09 (v01) 
        - Fuh-Cherng (Fuh) Jeng borrowed this script from Dr. Tzu-Hao (Harry) Lin, written in MATLAB
    2021-01-24 (v02)
         - Fuh rewrote this script in Python and made some minor adjustments
        
    '''

    import numpy as np
    from aeptools import corr_columnwise

    # reshape each spectrogram to a 1D array (i.e., flatten each spectrogram)
    signal = spectrogramSignal.reshape(-1, spectrogramSignal.shape[2], order='F')  

    # compute npermutation, nsweeplabels_unique, and nsweepCondition
    npermutation = np.sum(nsweeplabels == reference_sweeps)         
    nsweeplabels_unique = nsweeplabels[npermutation-1::npermutation]   
    nsweepCondition = len(nsweeplabels_unique)                       
    
    # create reference data
    ref_data = spectrogramData[:, :, nsweeplabels==reference_sweeps].reshape(-1, npermutation, order='F')           
    ref_data = np.mean(ref_data, axis=1).reshape(ref_data.shape[0],1)                         
    data = np.tile(ref_data, spectrogramData.shape[2])      

    # compute RMSE
    rmse = np.sqrt(np.mean((signal-data)**2, axis=0))    
    
    # compute CORR
    corr = corr_columnwise(signal, data)

    # reshape rmse and correlation results into 2D matrices (npermutation x nsweepCondition)
    rmse = rmse.reshape((npermutation, nsweepCondition), order='F')
    corr = corr.reshape((npermutation, nsweepCondition), order='F')
    
    # save results in RMSE and CORR dictionaries
    RMSE = {
        'results': rmse,
        'mean': np.mean(rmse, axis=0),   
        'std': np.std(rmse, axis=0),     
        'se': np.std(rmse, axis=0) / np.sqrt(npermutation)  
        }

    CORR = {
        'results': corr,
        'mean': np.mean(corr, axis=0),    
        'std': np.std(corr, axis=0),     
        'se': np.std(corr, axis=0) / np.sqrt(npermutation)  
        }
       
    return (RMSE, CORR, nsweeplabels_unique)


