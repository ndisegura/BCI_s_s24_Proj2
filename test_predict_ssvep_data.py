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
import predict_ssvep_data
import numpy as np

#Close previosly drawn plots
plt.close('all')

#Build data file string
data_directory = './SsvepData/'
subject=2

#%%
#PartA: Generate Predictions


#%%
#PartB: Calculate Accuracy and ITR  
#Use synthetic data and temp values for now.for now

#load data
data=import_ssvep_data.load_ssvep_data(subject,data_directory)
event_types=data['event_types']
#Create synthetic data

predicted_event=event_types.copy() #Start with an exact copy
#change 5 random entries
for i in range(10):
    rand_index=np.random.randint(0,19)
    #print(rand_index)
    #print(f'event types:{event_types}')
    #print(f'event predicted:{predicted_event}')
    predicted_event[i]=predicted_event[rand_index]

accuracy,ITR=predict_ssvep_data.get_accuracy_ITR(data,event_types,predicted_event)


#%%
#PartC: Loop Through Epoch Limits


#%%
#PartD: Plot results

#%%
#PartE: Create Predictor Histogram



