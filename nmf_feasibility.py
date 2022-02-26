# Research collaboration with Dr. Tzu-Hao (Harry) Lin
# 2019-08-13 (v01) 
#    - read in one EEG recording (raw data from .ss files), derive sub-averages and their spectrograms
# 2019-09-09 (v02) 
#    - Fuh included nsweeps in the filename of output spectrogram data. 
#    - Harry added codes for performance index
# 2020-10-22 (v03)
#    - Bre updates to accompany code for all dissertation participants 
#    - This code is modified as step _01 for Bre dissertation. 
#    - The purpose of this code is to read in .ss data, process, and output spectrogramData for each condition.
# 2021-10-01 (v04)
#    - rewrote this script in Python
# 20210-10-30 (v05)
#    - use a dictionary to provide input parameters for preprocess()   
# 2022-01-12 (v06) 
#    - read in 'data_epoch' directly that has been saved when running step 01preprocess
# 2022-01-26 (v07))
#    - read in ".ss" EEG raw data and process from there
#    - rewrite part of nmfsc() from Matlab to Python, and name it ssnmf()
# 2022-02-14 (v08)
#    - allow the user to use (1) custom ssnmf or (2) sklearn NMF class for decomposition

# ---- raw data ----
# EEG raw data are arranged in 2 dimension (npoints, nsweeps)

# For example, 150 ms (+ 45 ms), sampling rate 20,000 points/s, 10,000 sweeps

# Data structure: amplitude (nV), npoints = 20,000 npoints/s X 0.150 s (or 0.195) = 3000 points (or 3900 points), 10,000 sweeps

# Each recording has a large data size, e.g., (3900, 10,000) 

# --- spectrogram for each suba-average ---

# For example, if we randomly take 7,000 out of the first 8,000
# Another example, take 5,000 sweeps, but do it for only say 1000 times

# In other words, there will be a total of 1000 sub-averages for each EEG recording.
# When we look at the spectrogram of each sub-average,
# spectrogram specifications are
# X axis: 50 ms window size, 0.5 ms stepsize ==> 201 bins (X axis, time frame)
# Y axis: frequency from 1 to 1000 Hz, frequency resolution = 1 Hz
# Z axis: amplitude (nV) 

# spectrogramData is a 3D data structure: frequency x number of time frames x number of spectrograms (1000 x 201 x 55)
# Note: Number of spectrograms = number of permutations x number of nsweep conditions (e.g., 5 x 11 = 55)

# clear console, clear variables, reset kernel
from IPython import get_ipython         # The 'IPython' is a Python built-in module, for interactive computing in different languages
get_ipython().magic('clear')            # clear: clear the console 
get_ipython().magic('reset -f')         # -f : force reset without asking for confirmation (i.e., a 'hard' reset of the kernel)

# close all figures
import matplotlib.pyplot as plt         
plt.close('all')

# import modules
# from pathlib import Path                # The 'Path' class is a new, correct, and easy way to work with file paths and file names.
# from aeptools import nmf_evaluation    # file_to_par, preprocess, crossCorr, dataEpoch_to_spectrogramData,
# import sys                              # import the Python built-in module 'sys' for sys.exit(), sys.platform
# import pandas as pd                     # import Pandas for DataFrame
import numpy as np                      # import numpy
# import random
# import json
# import tkinter as tk
# from tkinter import filedialog
# import os
# from pathlib import Path
# from scipy.io  import loadmat
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from aeptools import nmf_decomposition_custom_ssnmf, nmf_evaluation_ssnmf
# from nmf_decomposition_sklearn_ssnmf import nmf_decomposition_sklearn_ssnmf

# initialize parameters
# HINT: After exeprimenting, the best parameters to use are: custom (instead of sklearn), Butterworth (insted of FIR), random (instead of sequential indices), nV (insted of log10 transformation)
# ssnmfDecompositionLibrary = 'custom'         # ['custom' or 'sklearn'], which open-source library to use for ssnmf decomposition
nbasis = 2                    # signal + noise
# appendReferenceFlag = False   # whether to append a reference spectrogram (e.g., a group-averaged spectrogram or a stimulus spectrogram), or not
# setRandomPermutationFlag = False                # use random permutaion numbers, or simply the first n sweeps
# setRandomSeedFlag = False
# outputSweepDataFlag = False 
# outputSpectrogramDataFlag = False
# outputDictParFlag = False                        # output a dictionary of parameters, or not
# processRawDataFlag = True
# readSpectrogramDataMatlabFlag = False           # whether to read spectrogramData directly from a Matlab script output, or not

nsweeps = np.array([100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])   # nsweeps is an array containing the numbers of sweeps that will be used to derive a subaveraged waveform
# nsweepCondition = len(nsweeps)
# npermutations = np.array([1]) # [1, 5, 10, 100]   # (npermutations) the number of runs of permutations that you want to repeat. 
# npermutationCondition = len(npermutations)
niter = 200    # number of iterations

# # select a file
# root = tk.Tk()     # create a handle for the root window of the file dialog
# root.withdraw()    # To show only the dialog without any other GUI elements, you have to hide the root window using the withdraw() method.
# root.update()      # This line is necessary for this script to work on macOS
# currentdir = os.getcwd()
# file_path = filedialog.askopenfilename(title='Select a file', filetypes=[("EEG data", ".ss"), ("All files", "*.*")], initialdir=currentdir)

# TODO: change all important dictionaries to classes, so that each new element can be added with a default value. Therefore, the old script can still work without modification.

# # specify a dictionary of relevant parameters
# dict_input_par_logbook = {
#     'experimentString' : 'FFR',
#     'lexicalTone' : 2, 
#     'groupString' : 'ENG', 
#     'AgeString' : 'newborn', 
#     'stimToken' : 'i2_150ms', 
#     'nRepetitions' : 1, 
#     'si' : 45, 
#     'dBSPL' : 70, 
#     'labviewIndex' : 1, 
#     'headerIndex' : 1, 
#     'nChans' : 1,
#     'hd' : 'S:',                                     # hard drive (hd)             
#     'logfilepath' : 'S:/data/GroupData_project/',    # including hd 
#     'logfilename' : 'FFR_log.npy',
#     'outputLogfileFlag' : False,
#     'randomSeeds' : [4, 1, 3, 0, 3]    
#     }

# # define a dictionary of input parameters for preprocess()
# dict_input_par_preprocess = {
#     'cutoff' : (90.0, 1500.0),                # cutoff frequencies in Hz
#     'filterMethod' : 1,                       # method for filtering the EEG time waveform [0 FIR, 1 butterworth [default = 1]]
#     'npoles' : 2,                             # number of poles [deafult = 500 for FIR, 4 for butterworth lfilter() and sosfilt(), 2 Butterworth filtfilt() and sosfiltfilt()]
#     'filterFunction' : 3,                     # filter function to use [0 lfilter(), 1 filtfilt(), 2 sosfilt(), 3 sosfiltfilt() [default = 3]]
#     'lowthr' : -25.0,                         # lower bound of artifact rejection criterion [default = -25.0]
#     'highthr' : 25.0,                         # upper bound of artifact rejection criterion [default = 25.0]
#     'nepochsTargeted' : 20000,                # targeted maximal number of epochs/sweeps to be included in the averaged waveform [default = 20000]
#     'takeOnlyTargetedNSweepsFlag' : True,     # take only targeted number of sweep or not [default = True]
#     'outputAvgDataFlag' : False,              # output avg or not [default = True]
#     'outputHeaderTailFlag' : False,           # output header and tail or not [default = True]
#     'outputRejectionIndexFlag' : False,       # output rejection index or not [default = True]
#     'outputFilterSettingsFlag' : False,       # output filter settings or not [default = True]
#     'outputPolarityDataFlag' : False,         # output polairty data or not [default = True]
#     'outputDictParFlag' : False,              # output dictionaries of important parameters or not [default = True]
#     'outputDataEpochFlag' : False,            # output data_epoch as a separate file or not [default = True]
#     }

# # define a dictionary of input parameters for crossCorr()
# dict_input_par_crossCorr = {
#     # data flags
#     'outputEntireDataFlag' : False,                 # output the entire length of the recordings (i.e., 150 ms + si)
#     'outputExtracted150msdataFlag' : False,         # output the 150-ms data segmented from the entire say 195-ms recording window
#     'outputPreStimDataFlag' : False,                # Flag to output pre-stimulus portion in a separate file
#     'outputLatAmpFlag' : False,                     # output latency and amplitude data, or not
#     'outputSpectrogramData_StimulusFlag' : False,   # output spectrogram data for stimulus, or not     
#     'outputSpectrogramData_RecordingFlag' : False,  # output spectrogram data for recording, or not   
#     'outputLatencyDataFlag' : False,                # output latency data, or not
#     'outputStimulusDataFlag' : False,               # output stimulus time waveform, or not  
    
#     # figure flags
#     'outputFigures_stimulusFlag' : False,           # output figures for stimulus, or not
#     'outputFigures_recordingFlag' : False,          # output figures for recording, or not
#     'setFigureLocationFlag' : True,                # set figure location, or not
#     'space' : 10,                                  # space between adjacent figures in pixels        
#     'dx' : 640,                                    # width of a figure in pixels
#     'dy' : 480,                                    # height of a figure in pixels
#     'colorMap_stimulus' : 1,                       # define color map for stimulus spectrogram {0: 'gray_r', 1: 'jet'}
#     'colorMap_recording' : 1,                      # define color map for recording spectrogram {0: 'gray_r', 1: 'jet'}
  
#     # spectrogram parameters
#     'win' : 50.0,                    # window size in ms for calculating spectrogram 
#     'gap' : 0.5,                     # step size in ms for calculating spectrogram [changed from 1 to 0.5 ms for 170 ms di1234 tokens]
#     'freqResolution' : 1.0,          # have a frequency resolution of 0.1 Hz here to increase accuracy in estimating f0 --- it will take a long time to run through, but the data will be worth it.
#     'upperFreq' : 1000.0,            # frequency range to take out from the spectrogram output [default 1000, i.e., take out 1-1000 Hz data out of a total of a 10,000 Hz data range
#     'freq_stepsize' : 10.0,         # frequency step size in Hz for plotting in Veusz
  
#     # other parameters
#     'polarityNames' : [''],          # ['', '_a', '_b']   # a list of polarity names to be included for analysis
#     'avg' : [],                      # averaged time waveform
#     'crossCorrFlag' : False,          # whether to perform cross-correlation, or not
#     'stimTokens' : [],               # stores a list of stimTokens that have been processed in this subroutine
#     }

# TODO: define a dictionary of input parameters for dataEpoch_to_spectrogramData()

# TODO: define a dictionary of input parameters for nmf decomposition
# outputDecompositionFigureFlag = True

# TODO: define a dictionary of input parameters for nmf evaluation
# outputEvaluatoinFigureFlag = True

# # get filepath, filename, filepre, etc.
# filePath = Path(file_path)
# directory = filePath.parts[-2]
# filepath = str(filePath.parents[0])
# filename = filePath.name
# filesuffix = filePath.suffix
# filepre = filePath.stem
# subjectCode = filename[0:6]
# sessionNumber = int(filename[filename.find('ses') + 3])
# fileNumberString = filename[11:14]
# hd = filePath._drv
# # conditionString = filepathstr + '_' + str(npermutation).zfill(3) + 'permutations'

# # specify logfilepath and logfilename
# logfilepath = filePath.parents[1] / 'processeData/'
# logfilename = 'NMF_feasibility_log.json'

# if processRawDataFlag:
    
#     # convert file string to par
#     par = file_to_par(file_path, dict_input_par_logbook)
    
#     # preprocess EEG raw data # NOTE: For this script, there is only one file in par. However, this loop is still necessary so that "file" is in a correct dtype inside the "for" loop.
#     for file in par.itertuples():     # loop through each file
#         (dict_output_par_preprocess, _, data_epoch) = preprocess(hd, logfilepath, logfilename, file, dict_input_par_preprocess)
    
#     # HINT: debug only
#     matlab_data_epoch4 = False
#     if matlab_data_epoch4:
#         data_epoch4 = loadmat('matlabData/data_epoch4.mat')['data_epoch']
#         np.all(data_epoch == data_epoch4)     # see if they are the same, or not       
#         np.allclose(data_epoch, data_epoch4)  # see if they are very close, or not ==> They are very close, but not the same. The max difference is about e-06 ==> It is neglectable.      
#         np.max(data_epoch - data_epoch4)      # find max difference ==> the max difference is about e-06 ==> neglectable
#         data_epoch0 = data_epoch              # swap data. Python --> data_epoch0.
#         data_epoch = data_epoch4              # Matlab --> data_epoch4 --> Python
#         del data_epoch4                       # del data to save memory

#     # convert data_epoch to spectrogramData
#     outputSpectrogramDataFlag = False
#     spectrogramData = dataEpoch_to_spectrogramData(data_epoch, dict_output_par_preprocess, dict_input_par_crossCorr, npermutationCondition, npermutations, nsweeps, outputSpectrogramDataFlag, filePath)
    
#     # HINT: debug only
#     matlab_spectrogramData = False
#     if matlab_spectrogramData:
#         spectrogramData1 = loadmat('matlabData/spectrogramData1.mat')['spectrogramData']
#         np.all(spectrogramData == spectrogramData1)     # see if they are the same, or not       
#         np.allclose(spectrogramData, spectrogramData1)  # see if they are very close, or not
#         np.max(spectrogramData - spectrogramData1)      # find max difference ==> the two spectrogramData look very similar ==> the max difference is big, but it is a result of random selection of data epoch. ==> if using sequential sweeps (or the same sweeps), the max difference is about 45 out of 1152 ==> about 4%
#         spectrogramData0 = spectrogramData              # swap data. Python --> spectrogramData0.
#         spectrogramData = spectrogramData1              # Matlab --> spectrogramData1 --> Python
#         # del spectrogramData1                            # del data to save memory

# else:
    
    # # read in spectrogramData
    # # spectrogramData is a 3D matrix (nfrequencies, nframes, nspectrograms), for example (1000, 201, 11)
    # # get filepath, filename, filepre, etc.
    # directory = filePath.parts[-2]
    # filepath = str(filePath.parents[0])
    # filename = filePath.name
    # filesuffix = filePath.suffix
    # filepre = filePath.stem
    # subjectCode = filename[0:6]
    # sessionNumber = int(filename[filename.find('ses') + 3])
    # fileNumberString = filename[11:14]
    # hd = filePath._drv
    # filepathstr = filepath + filepre + fileNumberString        
    # # conditionString = filepathstr + '_' + str(npermutation).zfill(3) + 'permutations'
    # # conditionString = str(filePath.parent + 
    # outputfilepre = filepre + '_001permutations'                                     
spectrogramData = np.load('spectrogram_data.npy', allow_pickle=True)

    # if readSpectrogramDataMatlabFlag:
    #     import h5py
        
    #     with h5py.File('./spectrogramData.mat') as f:
    #         arrays = {}
    #         for k, v in f.items():
    #             arrays[k] = np.array(v)
    #         spectrogramData = arrays['spectrogramData'].T
    #         del arrays
    
# --- perform NMF decomposition ---
# print('ssnmf decomposition library =', ssnmfDecompositionLibrary, flush=True)
# if ssnmfDecompositionLibrary == 'custom':
(spectrogramSignal, spectrogramNoise, W, H, nsweeplabels) = nmf_decomposition_custom_ssnmf(spectrogramData, nbasis, niter, nsweeps)
# elif ssnmfDecompositionLibrary == 'sklearn':
#     (spectrogramSignal, spectrogramNoise, W, H, nsweeplabels) = nmf_decomposition_sklearn_ssnmf(spectrogramData, nbasis, niter, appendReferenceFlag, npermutations, nsweeps, outputFigureFlag=outputDecompositionFigureFlag)
# else:
#     sys.exit('unrecognizable ssnmfDecompositionLibrary...')
    
# HINT: debug only
# NOTE: the separated spectrogramSignal and spectrogramNoise look similar, when plotted on an output figure. 
# matlab_output = False
# if matlab_output:
#     output1 = loadmat('matlabData/output1.mat')['output']
#     output1[np.isnan(output1)] = 0.0
    
#     ny = spectrogramData.shape[0]     # number of rows
#     nx = spectrogramData.shape[1]     # number of columns
#     nz = spectrogramData.shape[2]     # number of sheets
    
#     # separate output to be signal and noise
#     outputSignal1 = np.reshape(output1[:,:,0], (ny, -1), order='F')
#     outputNoise1 = np.reshape(output1[:,:,1], (ny, -1), order='F')
    
#     # reshape outputted signal and noise
#     spectrogramSignal1 = np.reshape(outputSignal1, (ny, nx, -1), order='F')  
#     spectrogramNoise1 = np.reshape(outputNoise1, (ny, nx, -1), order='F') 
      
#     np.all(spectrogramSignal == spectrogramSignal1)   # see if they are the same, or not       
#     np.all(spectrogramNoise == spectrogramNoise1)     # see if they are the same, or not       

#     spectrogramSignal0 = spectrogramSignal            # swap data. Python --> spectrogramSignal0.
#     spectrogramSignal = spectrogramSignal1            # Matlab --> spectrogramSignal1 --> Python
#     spectrogramNoise0 = spectrogramNoise              # swap data. Python --> spectrogramNoise0.
#     spectrogramNoise = spectrogramNoise1              # Matlab --> spectrogramNoise1 --> Python
#     # del spectrogramSignal1, spectrogramNoise1         # del data to save memory

#     # # TODO: make npermutations work for this script
#     # i = 0
#     npermutation = 1
    
#     # define a dictionary for converting numbers to color maps (for ploting spectrograms)
#     dict_colorMap = {
#         0 : 'gray_r',
#         1 : 'jet',
#         }
    
#     colorMap_recording = 1                        # define color map for recording spectrogram
    
#     # compute xticks and nsweeplabels
#     xticks = np.full(len(nsweeps) * npermutation, np.nan)
#     for j in range(len(nsweeps)):
#         for k in range(npermutation):
#             xticks[k+j*npermutation] = j*npermutation*nx + (k+1)*nx - 1
#     nsweeplabels = np.repeat(nsweeps, npermutation)
    
#     # compute xticks_unique and nsweeplabels_unique        
#     xticks_unique = xticks[npermutation-1::npermutation]             # for each run of npermutation, show only one tick label, to avoid overlaps 
#     nsweeplabels_unique = np.unique(nsweeplabels) 
    
#     # compute (left, right, bottom, top) boundaries for imshow(), when plotting original and separated spectrograms
#     left = -0.5              
#     right = nx*nz + 0.5
#     bottom = -0.5
#     top = ny + 0.5
            
#     data = np.reshape(spectrogramData, (spectrogramData.shape[0], -1), order='F') 
#     outputSignal = np.reshape(spectrogramSignal, (spectrogramSignal.shape[0], -1), order='F')
#     outputNoise = np.reshape(spectrogramNoise, (spectrogramNoise.shape[0], -1), order='F') 

#     # display the original data and source separation results
#     plt.rcParams["font.size"] = "10"    # set font size for all elements (titles, labels, ticks) for all subsequent matplotlib plots
#     fig106 = plt.figure(106, figsize=(8, 6))
#     ax1 = fig106.add_subplot(4,1,1)
#     im1 = ax1.imshow(data, extent=(left, right, bottom, top), origin='lower', aspect='auto', cmap=dict_colorMap.get(colorMap_recording))
#     ax1.set_title('Original spectrogram') 
#     ax1.set_ylabel('Frequency (Hz)')
#     ax1.set_xticks(xticks) 
#     ax1.set_xticklabels([])
#     divider1 = make_axes_locatable(ax1)
#     cax1 = divider1.append_axes('right', size='1%', pad=0.05)
#     fig106.colorbar(im1, cax=cax1, orientation='vertical')                      
#     ax2 = fig106.add_subplot(4,1,2)
#     xvec = np.arange(0, nx*nz, 1)
#     yvec = np.repeat(nsweeplabels, nx)
#     ax2.plot(xvec, yvec) 
#     ax2.set_title('Preparation of testing data') 
#     ax2.set_ylabel('N sweeps')        
#     ax2.set_xlim(min(xvec), max(xvec))
#     ax2.set_ylim(min(yvec), max(yvec))
#     ax2.set_xticks(xticks) 
#     ax2.set_xticklabels([])
#     ax2.locator_params(axis="y", nbins=4)      
#     ax3 = fig106.add_subplot(4,1,3)
#     im3 = ax3.imshow(outputSignal, extent=(left, right, bottom, top), origin='lower', aspect='auto', cmap=dict_colorMap.get(colorMap_recording))
#     ax3.set_title('Separated EEG signal') 
#     ax3.set_ylabel('Frequency (Hz)')
#     ax3.set_xticks(xticks) 
#     ax3.set_xticklabels([])
#     divider3 = make_axes_locatable(ax3)
#     cax3 = divider3.append_axes('right', size='1%', pad=0.05)
#     fig106.colorbar(im3, cax=cax3, orientation='vertical')        
#     ax4 = fig106.add_subplot(4,1,4)
#     im4 = ax4.imshow(outputNoise, extent=(left, right, bottom, top), origin='lower', aspect='auto', cmap=dict_colorMap.get(colorMap_recording))
#     ax4.set_title('Separated noise') 
#     ax4.set_xlabel('nSweep condition')
#     ax4.set_ylabel('Frequency (Hz)')
#     ax4.set_xticks(xticks_unique) 
#     ax4.set_xticklabels(nsweeplabels_unique)            
#     divider4 = make_axes_locatable(ax4)
#     cax4 = divider4.append_axes('right', size='1%', pad=0.05)
#     fig106.colorbar(im4, cax=cax4, orientation='vertical')        
#     plt.tight_layout()   # check and adjust the extents of ticklabels, axis labels, and titles, to avoid overlaps
#     plt.show(block=False)

# --- perform NMF evaluation ---
# npermutation = npermutations[0]
# nsweeplabels = np.repeat(nsweeps, npermutation)
reference_sweeps = 8000
nmf_evaluation_ssnmf(spectrogramData, spectrogramSignal, spectrogramNoise, reference_sweeps, nsweeps)
