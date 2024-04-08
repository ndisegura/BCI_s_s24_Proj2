"""
Andres Segura & Aiden Pricer-Coen
BME 6770: BCI's Project 02
Dr. David Jangraw
4/9/2024

Test scrip for the implementation of the predict_ssvep_data module. This script consist of 5 major sections.
The test module will load the SSVEP from the data dictionary and create a string array with the predicted labels.
figures of merit are then computed for different epoch start and end times,
"""

# Import the necessary modules
import os
import sys
import matplotlib.pyplot as plt
import import_ssvep_data
import filter_ssvep_data
import import_ssvep_data as imp
import predict_ssvep_data as prd

# Close previosly drawn plots
plt.close('all')

# Build data file string
data_directory = './SsvepData/'
subject = 1
#%%
# PartA: Generate Predictions

# Load subject data
data = imp.load_ssvep_data(subject, data_directory)

# Epoch subject data
epoch_start_time =0
epoch_end_time =20
#eeg_epochs, epoch_times, is_trial_15Hz = imp.epoch_ssvep_data(data,epoch_start_time,epoch_end_time)

# # Get epoched fft data
# fs = data['fs']
# eeg_epochs_fft, fft_frequencies = imp.get_frequency_spectrum(eeg_epochs, fs)

# Generate predicted labels
#Changed function argument so that we can reuse this function in part C. Also added "channel" argument. Need to fix functin to handle multiple channel. Using just one for now
predicted_labels = prd.generate_predicted_labels(data,epoch_start_time,epoch_end_time, 'Oz') 

print(predicted_labels)

#%%
# PartB: Calculate Accuracy and ITR

event_types=data['event_types']

accuracy,ITR=prd.get_accuracy_ITR(data,event_types,predicted_labels)
print(accuracy)
print(ITR)

#%%
# PartC: Loop Through Epoch Limits
accuracy_matrix,ITR_matrix,loop_epoch_time=prd.loop_epoch_limits(data, epoch_start_time_limit=5, epoch_end_time_limit=20, step=0.5, channel='Oz') 
print(accuracy_matrix)
print(ITR_matrix)

#%%
# PartD: Plot results

prd.generate_pseudocolor_plots(accuracy_matrix,ITR_matrix,loop_epoch_time)

#%%
# PartE: Create Predictor Histogram




