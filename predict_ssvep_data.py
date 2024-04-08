"""
@author: Andres Segura & Aiden Pricer-Coen
BME 6770: BCI's Lab 04
Dr. David Jangraw
4/9/2024

UPDATE Module Description:


This module provides functions to process Steady State Visual Evoked Potential signals.
The module will analyze the SSVEP data, generate predicted labels, compute accuracy and
information tranfer rate. The module also implements function to plot a confusion matrix
for accuracy and ITRgenerates, and predictor histograms.
"""

# Import Statements
from pylab import *
from scipy.signal import firwin, filtfilt, freqz, hilbert
import matplotlib.pyplot as plt
import numpy as np
import import_ssvep_data as imp

# PartA: Generate Predictions


def find_nearest_frequency_index(frequencies, target_frequency):
    """
    Find the index of the frequency closest to the target frequency.

    Parameters:
        frequencies (array): Array of frequencies.
        target_frequency (float): Target frequency.

    Returns:
        int: Index of the frequency closest to the target frequency.
    """
    # Find the index of the frequency closest to the target frequency
    return np.argmin(np.abs(frequencies - target_frequency))


def generate_predicted_labels(data,epoch_start_time,epoch_end_time, channel='Oz'):#eeg_epochs_fft, fft_frequencies):
    """
    Generate predicted labels for SSVEP data. The function will epoch the SSVEP data based on the start and end time 
    specified in the function arguments, compute the FFT spectrum of the data across channels, and will compare 
    the amplitude of frequencies at 12hz and 15hz for the selected channel. The highest amplitude is selected as the predicted label.

    Parameters:
        data (dict): Dictionary containing EEG data.
        epoch_start_time (float): Start time of the epoch in seconds.
        epoch_end_time (float): End time of the epoch in seconds.
        channel (str, optional): Name of the EEG channel. Defaults to 'Oz'.

    Returns:
        predicted_labels: List of predicted labels of size T (number of trials)
    """
    
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
    """
    Function to calculate accuracy and Information Transfer Rate (ITR).
    

    Parameters:
        data (dict): Dictionary containing EEG data.
        event_types (numpy array of str of size C, the number of channels): Array of true event types.
        predicted_event (numpy array of str of size C): Array of predicted event types.

    Returns:
        float: Accuracy.
        float: Information Transfer Rate (ITR).
    """
    
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
    """
    Loop through different epoch start and end times to compute accuracy and ITR.

    Parameters:
        data (dict): Dictionary containing EEG data.
        epoch_start_time_limit (float, optional): Lower limit of epoch start time in seconds. Defaults to 0.
        epoch_end_time_limit (float, optional): Upper limit of epoch end time in seconds. Defaults to 20.
        step (float, optional): Step size for epoch start and end times. Defaults to 0.1.
        channel (str, optional): Name of the EEG channel. Defaults to 'Oz'.

    Returns:
        accuracy_matrix (numpy array of size K x K ): Matrix of accuracy values.
        ITR_matrix (numpy array of size K x K): Matrix of ITR values.
        loop_epoch_time (numpy array of size 1xN): Array of epoch times.
    """
    
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
            
    return accuracy_matrix[::-1],ITR_matrix[::-1],loop_epoch_time


def generate_pseudocolor_plots(accuracy_matrix,ITR_matrix,loop_epoch_time):
    """
    Generate pseudocolor plots for accuracy and ITR at various epoch limits

    Parameters:
        accuracy_matrix (numpy array of size K x K ): Matrix of accuracy values.
        ITR_matrix (numpy array of size K x K): Matrix of ITR values.
        loop_epoch_time (numpy array of size 1xN): Array of epoch times.

    Returns:
        None
    """
    #Create figure and subplot handles
    fig, (ax0,ax1) = plt.subplots(1,2)
    alpha_val = 0.8
    
    
    ax0.set_title('Accuracy')
    accuracy_plot=ax0.imshow(accuracy_matrix*100,alpha=alpha_val,aspect=1, extent=[loop_epoch_time[0], loop_epoch_time[-1],loop_epoch_time[0], loop_epoch_time[-1]])
    #cax = fig.add_axes([ax0.get_position().x1+0.01,ax0.get_position().y0,0.02,ax0.get_position().height])
    cbar=fig.colorbar(accuracy_plot,ax=ax0,location='right',shrink=0.5)
    ax0.set_xlabel('Epoch end time [s]')
    ax0.set_ylabel('Epoch start time [s]')
    cbar.ax.set_ylabel('% correct', rotation=270)
    
    ax1.set_title('ITR')
    ITR_plot=ax1.imshow(ITR_matrix,alpha=alpha_val,aspect=1, extent=[loop_epoch_time[0], loop_epoch_time[-1],loop_epoch_time[0], loop_epoch_time[-1]])
    cbar=fig.colorbar(ITR_plot,ax=ax1,location='right',shrink=0.5)
    ax1.set_xlabel('Epoch end time [s]')
    ax1.set_ylabel('Epoch start time [s]')
    cbar.ax.set_ylabel('Information Transfer Rate (bits/s)', rotation=270)
    plt.tight_layout()
    
    return


def plot_predictor_histogram(data, epoch_start_time, epoch_end_time, channel='Oz'):
    """
    Plot predictor histogram showing the distribution of predictor values for each trial type.

    Parameters
    ----------
    data : dict
        Dictionary containing EEG data.
    epoch_start_time : float
        Start time of the epoch in seconds.
    epoch_end_time : float
        End time of the epoch in seconds.
    channel : str, optional
        Name of the EEG channel. Default is 'Oz'.
    """
    # Epoch Data
    eeg_epochs, _, _ = imp.epoch_ssvep_data(data, epoch_start_time, epoch_end_time)

    # Get epoched fft data
    fs = data['fs']
    eeg_epochs_fft, fft_frequencies = imp.get_frequency_spectrum(eeg_epochs, fs)

    # Find indices of target frequencies
    index_12hz = np.argmin(np.abs(fft_frequencies - 12))
    index_15hz = np.argmin(np.abs(fft_frequencies - 15))

    # Select channel
    channel_index = np.where(data['channels'] == channel)[0][0]

    # Separate epochs based on predicted labels
    predicted_labels = generate_predicted_labels(data, epoch_start_time, epoch_end_time, channel)
    predictor_12hz = []
    predictor_15hz = []

    for i, label in enumerate(predicted_labels):
        predictor_value = np.abs(eeg_epochs_fft[i, channel_index, index_15hz] - eeg_epochs_fft[i, channel_index,
                                                                                               index_12hz])
        if label == '12hz':
            predictor_12hz.append(predictor_value)
        else:
            predictor_15hz.append(predictor_value)

    # Plot histograms for each trial type
    plt.figure()
    plt.hist(predictor_12hz, histtype='barstacked', bins=20, color='c', edgecolor='black', label='12Hz Trials')
    plt.hist(predictor_15hz, histtype='barstacked', bins=20, color='m', edgecolor='black', label='15Hz Trials')
    plt.title('Predictor Histogram')
    plt.xlabel('Predictor Variable')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    
 
    
