```python
from pylab import *
from scipy.signal import firwin, filtfilt, freqz, hilbert
import matplotlib.pyplot as plt
import numpy as np
import import_ssvep_data as imp

def find_nearest_frequency_index(frequencies, target_frequency):
    """
    Find the index of the frequency closest to the target frequency.

    Parameters
    ----------
    frequencies : array_like
        Array of frequencies.
    target_frequency : float
        Target frequency.

    Returns
    -------
    int
        Index of the frequency closest to the target frequency.
    """
    return np.argmin(np.abs(frequencies - target_frequency))

def generate_predicted_labels(data, epoch_start_time, epoch_end_time, channel='Oz'):
    """
    Generate predicted labels based on SSVEP data.

    Parameters
    ----------
    data : dict
        Dictionary containing SSVEP data.
    epoch_start_time : int
        Start time (in seconds) of the epoch relative to the initial event.
    epoch_end_time : int
        End time (in seconds) of the epoch relative to the initial event.
    channel : str, optional
        Name of the EEG channel. Default is 'Oz'.

    Returns
    -------
    list
        List of predicted labels.
    """
    # Epoch Data first
    eeg_epochs, epoch_times, is_trial_15Hz = imp.epoch_ssvep_data(data, epoch_start_time, epoch_end_time)
    # Get epoched fft data
    fs = data['fs']
    eeg_epochs_fft, fft_frequencies = imp.get_frequency_spectrum(eeg_epochs, fs)
    eeg_channel = data['channels']
    is_channel_selected = eeg_channel == channel

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

def get_accuracy_ITR(data, event_types, predicted_event):
    """
    Compute accuracy and Information Transfer Rate (ITR) from SSVEP data.

    Parameters
    ----------
    data : dict
        Dictionary containing SSVEP data.
    event_types : array_like
        Array of event types.
    predicted_event : array_like
        Array of predicted event types.

    Returns
    -------
    tuple
        Tuple containing accuracy and ITR.
    """
    N = 2  # There are only 2 classes
    # Compute accuracy
    P = (event_types == predicted_event).sum() / len(predicted_event)
    if P == 1.0: P = 0.99999  # correct P so that ITR doesn't blow up
    ITR_trial = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))  # bits per trial

    # Compute overall time from Data
    eeg = data['eeg']
    fs = data['fs']
    trial_time = len(eeg[0]) * 1 / fs
    trial_count = len(predicted_event)
    ITR_second = ITR_trial * trial_count / trial_time
    return P, ITR_second

def loop_epoch_limits(data, epoch_start_time_limit=0, epoch_end_time_limit=20, step=0.1, channel='Oz'):
    """
    Loop through different epoch limits and compute accuracy and ITR.

    Parameters
    ----------
    data : dict
        Dictionary containing SSVEP data.
    epoch_start_time_limit : int, optional
        Start time limit for epochs (default is 0).
    epoch_end_time_limit : int, optional
        End time limit for epochs (default is 20).
    step : float, optional
        Step size for epoch limits (default is 0.1).
    channel : str, optional
        Name of the EEG channel (default is 'Oz').

    Returns
    -------
    tuple
        Tuple containing accuracy matrix, ITR matrix, and looped epoch times.
    """
    matrix_size = int((epoch_end_time_limit - epoch_start_time_limit) / step)
    accuracy_matrix = np.zeros(shape=(matrix_size, matrix_size), dtype=float)
    ITR_matrix = np.zeros(shape=(matrix_size, matrix_size), dtype=float)
    loop_epoch_time = np.arange(epoch_start_time_limit, epoch_end_time_limit, step)
    event_types = data['event_types']

    for matrix_row_index in range(matrix_size):
        for matrix_column_index in range(matrix_size):
            epoch_start_time = loop_epoch_time[matrix_row_index]
            epoch_end_time = loop_epoch_time[matrix_column_index]
            if epoch_end_time <= epoch_start_time:
                # Make sure we don't compute end times before start time
                P = 0.5
                ITR_second = 0.0
                accuracy_matrix[matrix_row_index, matrix_column_index] = 0.5
                ITR_matrix[matrix_row_index, matrix_column_index] = 0.00
            else:
                # Compute figures of merit
                predicted_labels = generate_predicted_labels(data, epoch_start_time, epoch_end_time, channel)
                P, ITR_second = get_accuracy_ITR(data, event_types, predicted_labels)
                # Update Accuracy and ITR matrices
                accuracy_matrix[matrix_row_index, matrix_column_index] = P
                ITR_matrix[matrix_row_index, matrix_column_index] = ITR_second

    return accuracy_matrix[::-1], ITR_matrix[::-1], loop_epoch_time

def generate_pseudocolor_plots(accuracy_matrix, ITR_matrix, loop_epoch_time):
    """
    Generate pseudocolor plots for accuracy and ITR.

    Parameters
    ----------
    accuracy_matrix : array_like
        Matrix containing accuracy values.
    ITR_matrix : array_like
        Matrix containing ITR values.
    loop_epoch_time : array_like
        Array of looped epoch times.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2)
    alpha_val = 0.8

    ax0.set_title('Accuracy')
    accuracy_plot = ax0.imshow(accuracy_matrix * 100, alpha=alpha_val, aspect=1,
                               extent=[loop_epoch_time[0], loop_epoch_time[-1], loop_epoch_time[0],
                                       loop_epoch_time[-1]])
    # cax = fig.add_axes([ax0.get_position().x1+0.01,ax0.get_position().y0,0.02,ax0.get_position().height])
    cbar = fig.colorbar(accuracy_plot, ax=ax0, location='right', shrink=0.5)
    # cbar=plt.colorbar(accuracy_plot,ax=cax)
    ax0.set_xlabel('Epoch end time [s]')
    ax0.set_ylabel('Epoch start time [s]')
    cbar.ax.set_ylabel('% correct', rotation=270)

    ax1.set_title('ITR')
    ITR_plot = ax1.imshow(ITR_matrix, alpha=alpha_val, aspect=1,
                          extent=[loop_epoch_time[0], loop_epoch_time[-1], loop_epoch_time[0], loop_epoch_time[-1]])
    # cax = fig.add_axes([ax1.get_position().x1+0.01,ax1.get_position().y0,0.02,ax1.get_position().height])
    # cbar=fig.colorbar(ITR_plot,ax=cax)
    cbar = fig.colorbar(ITR_plot, ax=ax1, location='right', shrink=0.5)
    ax1.set_xlabel('Epoch end time [s]')
    ax1.set_ylabel('Epoch start time [s]')
    cbar.ax.set_ylabel('ITR (bits/s)', rotation=270)
    plt.tight_layout()
    plt.show()


def plot_predictor_histogram(data, epoch_start_time, epoch_end_time, channel='Oz'):
    """

    Parameters
    ----------
    data
    epoch_start_time
    epoch_end_time
    channel
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

    # Calculate predictor variable for each epoch
    predictor = np.abs(
        eeg_epochs_fft[:, channel_index, index_15hz] - eeg_epochs_fft[:, channel_index, index_12hz])

    plt.hist(predictor, bins=20, color='c', edgecolor='black')
    plt.title('Predictor Histogram')
    plt.xlabel('Predictor Variable')
    plt.ylabel('Frequency')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    
    
 
    
