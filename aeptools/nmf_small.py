import numpy as np
from tqdm import tqdm


def ssnmf_core(V, rdim, iter_num, W0, H0):
    if (V < 0).any():
        raise ValueError("Negative values in data!")

    vdim = V.shape[0]
    samples = V.shape[1]

    W = W0
    H = H0

    for iteration in range(1, iter_num + 1):

        # Update using standard NMF multiplicative update rule
        H = H * (W.T @ V) / (W.T @ W @ H + 1e-9)

        # Renormalize so rows of H have constant energy
        norms = np.sqrt(np.sum(H.T ** 2, 0))
        H = H / np.outer(norms.T, np.ones((1, samples)))
        W = W * np.outer(np.ones((vdim, 1)), norms)
    
        # Update using standard NMF multiplicative update rule
        W = W * (V @ H.T) / (W @ H @ H.T + 1e-9)

    return (W, H)


def ssnmf(spectrogramData2D, number_basis, number_iter): 
    '''
    2022-01-30 (v01)
        - Fuh-Cherng (Fuh) Jeng converted part of the nmfsc() (that was originally written by Tzu-Hao Lin in Matlab) to Python
        - Fuh made a few minor changes
    2022-02-22 (v02)
        - Fuh modified this script so that the input spectrogramData2D is already a 2D matrix. 
        - As such, no reshape() is needed within this function. 
        - In other words, any reshape() should be performed prior to feeding spectrogramData2D into this function.
    '''
  
    data = 20 * np.log10(spectrogramData2D)   # convert nV to dB (adopted from Lin et al., 2017)

    data[data < 0] = 0                        

    n = data.shape[1]
    weight = np.ones((1, n))

    W = abs(np.random.randn(data.shape[0], number_basis))
    H = abs(np.random.randn(number_basis, n))
    H = H / np.outer(np.sqrt(np.sum(H ** 2, 1)), np.ones((1, n)))

    # NMF decomposition
    print('performing ssnmf(), please wait...', flush=True)
    for run in tqdm(range(number_iter)):
        H[0, :] = weight * np.sum(H[0, :]) / np.sum(weight)
        (W, H) = ssnmf_core(data, number_basis, 1, W, H)

    output = np.zeros((data.shape[0], data.shape[1], number_basis))
    np.seterr(invalid='ignore')     
    for run in range(number_basis):
        output[:, :, run] = (
            spectrogramData2D * np.outer(W[:, run], H[run, :]) / (W @ H)
        ) 

    np.seterr(invalid='warn')   
    output[np.isnan(output)] = 0

    return (output, W, H)
