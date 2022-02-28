def nmf_evaluation_ssnmf(spectrogramData, spectrogramSignal, spectrogramNoise, reference_sweeps, nsweeps):

    import matplotlib.pyplot as plt         
    from ssnmftools import nmf_evaluation
    import numpy as np                     
    
    
    npermutation = 1                    # number of permutations to perform [default = 1]
    
    plt.rcParams['toolbar'] = 'None'       
       
    nx = spectrogramData.shape[1]       # number of columns
    
    # compute xticks and nsweeplabels
    xticks = np.full(len(nsweeps) * npermutation, np.nan)
    for j in range(len(nsweeps)):
        for k in range(npermutation):
            xticks[k+j*npermutation] = j*npermutation*nx + (k+1)*nx - 1
    nsweeplabels = np.repeat(nsweeps, npermutation)
         
    # evaluate NMF in terms of RMSE and correlation coefficients
    (ORI_RMSE, ORI_CORR, ORI_nsweeplabels_unique) = nmf_evaluation(spectrogramData, spectrogramData, nsweeplabels, reference_sweeps)
    (NMF_RMSE, NMF_CORR, NMF_nsweeplabels_unique) = nmf_evaluation(spectrogramSignal, spectrogramData, nsweeplabels, reference_sweeps)
    
    absCCFlag = True                    
    if absCCFlag:
        ORI_CORR['results'] = abs(ORI_CORR['results'])
        NMF_CORR['results'] = abs(NMF_CORR['results'])
            
    # compute net effects of NMF (i.e., difference = NMF - ORI)      
    dif_rmse = NMF_RMSE['results'] - ORI_RMSE['results']
    dif_corr = NMF_CORR['results'] - ORI_CORR['results']
    
    # save results in a dictionary
    DIF_RMSE = {
        'results': dif_rmse,
        'mean': np.mean(dif_rmse, axis=0),  
        'std': np.std(dif_rmse, axis=0),    
        'se': np.std(dif_rmse, axis=0) / np.sqrt(npermutation)   
        }
    
    DIF_CORR = {
        'results': dif_corr,
        'mean': np.mean(dif_corr, axis=0),   
        'std': np.std(dif_corr, axis=0),    
        'se': np.std(dif_corr, axis=0) / np.sqrt(npermutation)   
        }
    
    # display performance curves (NMF, ORI)
    plt.rcParams["font.size"] = "10"    
    fig201 = plt.figure(201, figsize=(8, 6))
    ax1 = fig201.add_subplot(1,2,1)
    ax1.plot(ORI_nsweeplabels_unique, ORI_RMSE['mean'], label='AVG', marker='o', markersize=5)  
    ax1.plot(NMF_nsweeplabels_unique, NMF_RMSE['mean'], label='AVG+NMF', marker='^', markersize=5) 
    ax1.set_xlabel('Number of sweeps')
    ax1.set_ylabel('Root-mean-square error (RMSE)')
    ax1.set_ylim(-2.0, 25.0)
    ax1.legend(loc='upper right')
    ax2 = fig201.add_subplot(1,2,2)
    ax2.plot(ORI_nsweeplabels_unique, ORI_CORR['mean'], label='AVG', marker='o', markersize=5)
    ax2.plot(NMF_nsweeplabels_unique, NMF_CORR['mean'], label='AVG+NMF', marker='^', markersize=5)
    ax2.set_xlabel('Number of sweeps')
    ax2.set_ylabel('Correlation Coefficient (CC)')
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc='lower right')
    plt.tight_layout()  
    plt.show(block=False)
    
    # display performance curves (DIF = net effects of NMF)
    plt.rcParams["font.size"] = "10"    
    fig202 = plt.figure(202, figsize=(8,6))
    ax3 = fig202.add_subplot(1,2,1)
    ax3.plot(nsweeps, DIF_RMSE['mean'], label='Performance', marker='o', markersize=5)
    ax3.set_xlabel('Number of sweeps')
    ax3.set_ylabel('RMSE difference (AVG+NMF minus AVG)')
    ax3.legend(loc='lower right')
    ax4 = fig202.add_subplot(1,2,2)
    ax4.plot(nsweeps, DIF_CORR['mean'], label='Performance', marker='o', markersize=5)
    ax4.set_xlabel('Number of sweeps')
    ax4.set_ylabel('CC difference (AVG+NMF minus AVG)')
    ax4.legend(loc='upper right')
    plt.tight_layout()   
    plt.show(block=False)
    