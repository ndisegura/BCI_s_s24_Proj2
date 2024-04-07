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

# Import Statements
from pylab import *
from scipy.signal import firwin, filtfilt, freqz, hilbert
import matplotlib.pyplot as plt
import numpy as np
import import_ssvep_data as imp

# PartA: Generate Predictions


def find_nearest_frequency_index(frequencies, target_frequency):
    # Find the index of the frequency closest to the target frequency
    return np.argmin(np.abs(frequencies - target_frequency))


def generate_predicted_labels(data,epoch_start_time,epoch_end_time, channel='Oz'):#eeg_epochs_fft, fft_frequencies):
    
    #Epoch Data first
    eeg_epochs, epoch_times, is_trial_15Hz = imp.epoch_ssvep_data(data,epoch_start_time,epoch_end_time)
    # Get epoched fft data
    fs = data['fs']
    eeg_epochs_fft, fft_frequencies = imp.get_frequency_spectrum(eeg_epochs, fs)
    eeg_channel=data['channels']
    is_channel_selected=eeg_channel==channel

    
    predicted_labels = []

    # Define the target frequencies
    target_frequencies = [12, 15]

    # Iterate through epochs
    for epoch_fft in eeg_epochs_fft:
        # Initialize variables to store amplitudes at target frequencies
        amplitudes_at_target_freqs = []

        # Iterate through target frequencies
        for target_freq in target_frequencies:
            # Find the index of the frequency closest to the target frequency
            nearest_freq_index = find_nearest_frequency_index(fft_frequencies, target_freq)

            # Get the amplitude at the nearest frequency index AND selected channel
            amplitude_at_target_freq = np.absolute(epoch_fft[is_channel_selected, nearest_freq_index])

            # Append the amplitude to the list
            amplitudes_at_target_freqs.append(amplitude_at_target_freq)

        # Select the predicted frequency based on the higher amplitude
        if any(amplitudes_at_target_freqs[0] > amplitudes_at_target_freqs[1]):
            predicted_label = '12hz'
        else:
            predicted_label = '15hz'

        # Append the predicted label to the list
        predicted_labels.append(predicted_label)

    return predicted_labels

def get_accuracy_ITR(data,event_types,predicted_event):
    
    N=2 #There are only 2 clases
    #Compute accuracy
    P=(event_types==predicted_event).sum()/len(predicted_event)
    if P==1.0:P=0.99999 #correct P so that ITR doesn't blow up
    ITR_trial=np.log2(N) + P* np.log2(P) + (1-P) * np.log2 ((1-P)/(N-1)) #bits per trial
    
    #Compute overall time from Data 
    eeg=data['eeg']
    fs=data['fs']
    trial_time=len(eeg[0])*1/fs
    trial_count=len(predicted_event)
    ITR_second=ITR_trial*trial_count/trial_time
    return P,ITR_second

def loop_epoch_limits(data,epoch_start_time_limit=0, epoch_end_time_limit=20, step=0.1, channel='Oz'):
    
    matrix_size=int((epoch_end_time_limit-epoch_start_time_limit)/step)
    accuracy_matrix=np.zeros(shape=(matrix_size,matrix_size),dtype=float)
    ITR_matrix=np.zeros(shape=(matrix_size,matrix_size),dtype=float)
    loop_epoch_time=np.arange(epoch_start_time_limit,epoch_end_time_limit,step)
    event_types=data['event_types']
    
    for matrix_row_index in range(matrix_size):
        for matrix_column_index in range(matrix_size):
            
            epoch_start_time=loop_epoch_time[matrix_row_index]
            epoch_end_time=loop_epoch_time[matrix_column_index]
            #print(f'X-time:{epoch_start_time} Y-Time:{epoch_end_time}\n')
            if epoch_end_time <= epoch_start_time:
                #Make sure we don't compute end times before start time
                P=0.5
                ITR_second=0.0
                accuracy_matrix[matrix_row_index,matrix_column_index]=0.5
                ITR_matrix[matrix_row_index,matrix_column_index]=0.00
                
            else:
                #Compute figures of merit
                predicted_labels=generate_predicted_labels(data,epoch_start_time,epoch_end_time, channel)
                #print(predicted_labels)
                P,ITR_second=get_accuracy_ITR(data,event_types,predicted_labels)
                #Update Accuracy and ITR matrices
                accuracy_matrix[matrix_row_index,matrix_column_index]=P
                ITR_matrix[matrix_row_index,matrix_column_index]=ITR_second
            
            #print(f'accuracy:{P}, ITR:{ITR_second}\n')
            
    return accuracy_matrix[::-1],ITR_matrix[::-1]
    
    
    
 
    