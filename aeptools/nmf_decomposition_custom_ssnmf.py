def nmf_decomposition_custom_ssnmf(spectrogramData, nbasis, niter, nsweeps):
    '''
    Purpose : perform SSNMF decomposition
    
    Introduction
    ------------
    - corr is the Pearson correlation coefficient between spectrogramSignal and spectrogramData
    - RMSE (root-mean-square error) is the RMS value of the differences between spectrogramSignal and spectrogramData
    - NOTE: When estimating Correlation and RMSE, computations are based on "straightened/flattened spectrogram data". That is, 2D spectrograms are reshaped into vectors, prior to computing differnces and correlations.

    Parameters
    ----------
    spectrogramData : numpy array (3D)
        - spectrogramData contains a series of spectrograms. It is a 3 dimensional matrix: number of frequencies x number of time frames x number of spectrograms (e.g., 1000 x 201 x 11)
    nbasis : integer
        - number of basis
    niter : integer
        - number of iterations
    nsweeps : numpy array (1D) of integers
        - number of sweeps included for each spectrogram
 
   Returns
   -------
   spectrogramSignal : numpy array (3D)
       - spectrogramSignal is the FFR extracted from the SSNMF algorithm. It is a 3 dimensional matrix: number of frequencies x number of time frames x number of spectrograms (e.g., 1000 x 201 x 11)
   spectrogramNoise : numpy array (3D)
       - spectrogramNoise is the noisr extracted from the SSNMF algorithm. It is a 3 dimensional matrix: number of frequencies x number of time frames x number of spectrograms (e.g., 1000 x 201 x 11)
   W : numpy array (2D)
       - W is the spectral-basis matrix. It is a 2 dimenational matrix (e.g., 201000 x 2)
   H : numy array (2D)
       - H is the information-coding matrix. It is a 2 dimensional matrix (e.g., 2 x 11)
   nsweeplabels : numy array (1D)
        - nsweeplabels are the nSweeps labels that corresponds to all the spectrograms saved in spectrogramData (and should be the same to those in spectrogramSignal as well) 
   
    Diary
    -----    
    2020-12-31 (v01) 
        - Fuh-Cherng (Fuh) Jeng wrote this script
    2021-08-08 (v02) 
        - Fuh made some minor changes
    2022-02-22 (v03)
        - The user is required to reshape the input spectrogramData into a 2D matrix, prior to feeding it into the ssnmf() function. 
        - The benefit it that, starting from this version, the original spectrogramData can be reshaped in any way. As long as the final spectrogramData2D is a 2D matrix, the ssnmf() will work as it should be.        
    '''
    
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from aeptools import ssnmf
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if spectrogramData.size == 0:
        sys.exit('spectrogramData is empty...')

    # re-arrange spectrogramData to a 2D matrix 
    spectrogramData2D = spectrogramData.reshape((-1, spectrogramData.shape[2]), order="F")

    # --- perform ssnmf() ---
    (output, W, H) = ssnmf(spectrogramData2D, nbasis, niter) 

    data = np.reshape(spectrogramData, (spectrogramData.shape[0], -1), order='F')

    ny = spectrogramData.shape[0]     # number of rows
    nx = spectrogramData.shape[1]     # number of columns
    nz = spectrogramData.shape[2]     # number of sheets

    # force NaN to zero
    output[np.isnan(output)] = 0.0

    # separate output to be signal and noise
    outputSignal = np.reshape(output[:,:,0], (ny, -1), order='F')
    outputNoise = np.reshape(output[:,:,1], (ny, -1), order='F')

    # reshape outputted signal and noise
    spectrogramSignal = np.reshape(outputSignal, (ny, nx, -1), order='F')
    spectrogramNoise = np.reshape(outputNoise, (ny, nx, -1), order='F')

    npermutation = 1

    # compute xticks and nsweeplabels
    xticks = np.full(len(nsweeps) * npermutation, np.nan)
    for j in range(len(nsweeps)):
        for k in range(npermutation):
            xticks[k+j*npermutation] = j*npermutation*nx + (k+1)*nx - 1
    nsweeplabels = np.repeat(nsweeps, npermutation)

    # compute xticks_unique and nsweeplabels_unique
    xticks_unique = xticks[npermutation-1::npermutation]             
    nsweeplabels_unique = np.unique(nsweeplabels)

    # compute (left, right, bottom, top) boundaries for imshow() plots
    left = -0.5
    right = nx*nz + 0.5
    bottom = -0.5
    top = ny + 0.5

    # display the original data and source separation results
    plt.rcParams["font.size"] = "10"    
    fig101 = plt.figure(101, figsize=(8, 6))
    ax1 = fig101.add_subplot(4,1,1)
    im1 = ax1.imshow(data, extent=(left, right, bottom, top), origin='lower', aspect='auto', cmap='jet', vmin=0.0, vmax=60.0)
    ax1.set_title('Original spectrogram')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='1%', pad=0.05)
    fig101.colorbar(im1, cax=cax1, orientation='vertical')
    ax2 = fig101.add_subplot(4,1,2)
    xvec = np.arange(0, nx*nz, 1)
    yvec = np.repeat(nsweeplabels, nx)
    ax2.plot(xvec, yvec)
    ax2.set_title('Preparation of testing data')
    ax2.set_ylabel('N sweeps')
    ax2.set_xlim(min(xvec), max(xvec))
    ax2.set_ylim(min(yvec), max(yvec))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([])
    ax2.locator_params(axis="y", nbins=4)
    ax3 = fig101.add_subplot(4,1,3)
    im3 = ax3.imshow(outputSignal, extent=(left, right, bottom, top), origin='lower', aspect='auto', cmap='jet', vmin=0.0, vmax=60.0)
    ax3.set_title('Separated EEG signal')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xticks(xticks)
    ax3.set_xticklabels([])
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='1%', pad=0.05)
    fig101.colorbar(im3, cax=cax3, orientation='vertical')
    ax4 = fig101.add_subplot(4,1,4)
    im4 = ax4.imshow(outputNoise, extent=(left, right, bottom, top), origin='lower', aspect='auto', cmap='jet', vmin=0.0, vmax=60.0)
    ax4.set_title('Separated noise')
    ax4.set_xlabel('nSweep condition')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_xticks(xticks_unique)
    ax4.set_xticklabels(nsweeplabels_unique)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='1%', pad=0.05)
    fig101.colorbar(im4, cax=cax4, orientation='vertical')
    plt.tight_layout()   
    plt.show(block=False)

    return (spectrogramSignal, spectrogramNoise, W, H, nsweeplabels)
