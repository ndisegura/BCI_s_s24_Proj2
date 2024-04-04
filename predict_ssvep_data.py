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
import import_ssvep_data

# PartA: Generate Predictions


def find_nearest_frequency_index(frequencies, target_frequency):
    # Find the index of the frequency closest to the target frequency
    return np.argmin(np.abs(frequencies - target_frequency))


def generate_predicted_labels(eeg_epochs_fft, fft_frequencies):
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

            # Get the amplitude at the nearest frequency index
            amplitude_at_target_freq = epoch_fft[:, nearest_freq_index]

            # Append the amplitude to the list
            amplitudes_at_target_freqs.append(amplitude_at_target_freq)

        # Select the predicted frequency based on the higher amplitude
        if any(amplitudes_at_target_freqs[0] > amplitudes_at_target_freqs[1]):
            predicted_label = 12
        else:
            predicted_label = 15

        # Append the predicted label to the list
        predicted_labels.append(predicted_label)

    return predicted_labels

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
    