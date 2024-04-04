# -*- coding: utf-8 -*-
"""

@author: Andres Segura & Aiden Pricer-Coen
BME 6770: BCI's Lab 04
Dr. David Jangraw
4/9/2024

UPDATE Module Description:
    
example description:    
This module provides functions to process Steady State Visual Evoked Potential signals.
The module generates FIR filter taps, convolves (filters) the selected data and plots the signals
at the various stages
"""

from pylab import *
from scipy.signal import firwin, filtfilt,freqz,hilbert
import matplotlib.pyplot as plt
import numpy as np
import import_ssvep_data


def get_accuracy_ITR(data,event_types,predicted_event):
    
    N=2 #There are only 2 clases
    #Compute accuracy
    P=(event_types==predicted_event).sum()/len(predicted_event)
    ITR_trial=np.log2(N) + P* np.log2(P) + (1-P) * np.log2 ((1-P)/(N-1)) #bits per trial
    
    #Compute overall time from Data 
    eeg=data['eeg']
    fs=data['fs']
    trial_time=len(eeg[0])*1/fs
    trial_count=len(predicted_event)
    ITR_second=ITR_trial*trial_count/trial_time
    return P,ITR_second
    