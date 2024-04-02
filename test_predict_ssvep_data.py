# -*- coding: utf-8 -*-
"""
Andres Segura & Aiden Pricer_Coen
BME 6770: BCI's Project 02
Dr. David Jangraw
4/9/2024

Test scrip for the implementation of the predict_ssvep_data module. This script ...

e.g. description of test module:
loads the data from the data dictionary, 
followed by the design of two bandpass FIR filters centered around 12hz and 15hz. The filter taps are then used to convolve
it with the EEG data contained in the "data" dictionary. A Hilbert transform extracts the envelope of the oscillatory signals, then
the envelope and the event types are ploted on a single grapth. As a final step, the power spectrum of the raw EEG , filtered and 
envelope is ploted for the 'Oz" and 'Fz' EEG channels.
"""


# Inport the necessry modules
import os
import sys
import matplotlib.pyplot as plt
import import_ssvep_data
import filter_ssvep_data

#Close previosly drawn plots
plt.close('all')

#Build data file string
data_directory = './SsvepData/'
subject=2

#%%
#PartA: Generate Predictions


#%%
#PartB: Calculate Accuracy and ITR  


#%%
#PartC: Loop Through Epoch Limits


#%%
#PartD: Plot results

#%%
#PartE: Create Predictor Histogram



