# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:24:41 2024

@author: asegura2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def load_ssvep_data(subject,data_directory):
    '''
    This function loads ssvep data and stores it in a dictionary for easy 
    access.

    Parameters
    ----------
    subject : int
        subject number of data to load in
    data_directory :str
        path to data folder.

    Returns
    -------
    data_dict : dictonary 
         containing feilds of ssvep data representing strings, floats, and bool, 
         size of dictionary 1 x N where N is the amount of feilds

    '''
    data_file=f'{data_directory}SSVEP_S{subject}.npz'
    
    # Load dictionary
    data_dict = np.load(data_file,allow_pickle=True)
    
    return data_dict

    

def plot_raw_data(data,subject,channels_to_plot):
    '''
    Plots raw ssvep data 

    Parameters
    ----------
    data : dictonary 
         containing feilds of ssvep data representing strings, floats, and bool, 
         size of dictionary N x 1 where N is the amount of feilds
    subject : int
        subject number of data to load in
    channels_to_plot : list of str
        list of spatial electrode locations to be plotted, size N x 1 where n
        is the amount of spatial electrodes

    Returns
    -------
    None.

    '''

    channels=data['channels']
    eeg=data['eeg']
    fs=data['fs']
    eeg_time=np.arange(0,len(eeg[0])*1/fs,1/fs)
    
    event_samples=data['event_samples']
    event_duration=data['event_durations']
    event_type=data['event_types']
    
    is_channel_match=np.zeros(len(eeg[0]),dtype=bool)
    
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle(f'SSVEP Subject {subject} Raw Data')
    
    #PLot Event types
    
    for event_index, event_freq in enumerate(event_type):
        start_time=eeg_time[event_samples[event_index]]
        end_time=eeg_time[event_samples[event_index]+int(event_duration[event_index])]
        axs[0].plot([start_time,end_time],[event_freq,event_freq], 'b')
    axs[0].set_ylabel('Flash Frequency')
    axs[0].set_xlabel('Time (s)')
    axs[0].grid()
        
    #PLot EEG Data
    for channel_index, channel_member in enumerate(channels_to_plot):
        
        is_channel_match=channels==channel_member #Boolean indexing across rows for item in list
        
        selected_channel_data=eeg[is_channel_match]/10-6 #Divide by 10-6 to obtain voltage in uV
        
        axs[1].plot(eeg_time, np.squeeze(selected_channel_data),label=channel_member)
    axs[1].set_ylabel('Voltage (uV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
        


#%%

def epoch_ssvep_data(data,epoch_start_time,epoch_end_time):
    '''
    Epochs ssvep data based on when the event occurs and up to 20 seconds after  

    Parameters
    ----------
    data : dictonary 
         containing feilds of ssvep data representing strings, floats, and bool, 
         size of dictionary N x 1 where N is the amount of feilds
    epoch_start_time : int
        Time in seconds before the event starts. The default is -.5.
    epoch_end_time : int
        Time in seconds after the event starts

    Returns
    -------
    eeg_epochs : 3D array of floats 
         Number of events x Number of channels x number of samples per epoch
    epoch_time :2D Array of floats 
        samples per epoch x 1 array of times relative to event onset.
    is_trial_15Hz : array of bool
        boolean array true if event occured at 15hz, size N x 1 where N is 
        number of events 

    '''
    
    channels=data['channels']
    eeg=data['eeg']
    fs=data['fs']
    eeg_time=np.arange(0,len(eeg[0])*1/fs,1/fs)

    event_samples=data['event_samples']
    event_duration=data['event_durations']
    event_type=data['event_types']

    channel_count= len(channels)
    event_count = len(event_samples)


    samples_per_second = int(1/(eeg_time[1]-eeg_time[0]))
    seconds_per_epoch = epoch_end_time-epoch_start_time
    samples_per_epoch = int(samples_per_second * seconds_per_epoch)

    #create empty epoch array
    eeg_epochs = np.zeros((event_count, channel_count, samples_per_epoch))

    for event_index, event in enumerate(event_samples):
        
        # get eeg data_dict within the timebounds of the event
        data_to_add = eeg[:,event:event+samples_per_epoch]
        
        # add eeg data_dict into epoch
        eeg_epochs[event_index,:,:] = data_to_add

    #get time relative to each event 
    epoch_time = eeg_time[:samples_per_epoch]
    
    #create boolean array true if 15hz flash during epoch
    is_trial_15Hz = event_type== '15hz'
  
    return(eeg_epochs,epoch_time,is_trial_15Hz)

def epoch_generic_data(data,epoch_start_time,epoch_end_time, event_samples,event_duration,event_type,fs):
    
    data_time=np.arange(0,len(data[0])*1/fs,1/fs)
    
    channel_count=data.shape[0]
    event_count = len(event_samples)
    
    
    samples_per_second = int(1/(data_time[1]-data_time[0]))
    seconds_per_epoch = epoch_end_time-epoch_start_time
    samples_per_epoch = int(samples_per_second * seconds_per_epoch)
    
    #Compute indexes offset for start and end times other than 0 to 20 seconds
    epoch_start_offset=int(epoch_start_time*fs)
    epoch_end_offset=int(epoch_end_time*fs)
    
    #create empty epoch array
    data_epochs = np.zeros((event_count, channel_count, samples_per_epoch))
    
    #loop through each event samples and build epoch array
    for event_index, event_sample in enumerate(event_samples):
        
        # get eeg data_dict within the timebounds of the event
        data_to_add = data[:,event_sample + epoch_start_offset : event_sample + epoch_end_offset]
        
        # add selected data into epoch
        data_epochs[event_index,:,:] = data_to_add

    #get time relative to each event 
    epoch_time = data_time[:samples_per_epoch]
    
    #create boolean array true if 15hz flash during epoch
    is_trial_15Hz = event_type== '15hz'

    return (data_epochs,epoch_time,is_trial_15Hz)
    
def get_frequency_spectrum(eeg_epochs,fs):
    """
    Function to compute the fast fourier transform of epoch'd eeg data'
    
    Parameters
    ----------
    eeg_epochs : numpy array of floats of size T x C x S where T is the number 
    of event TRIALS, C is the number of channels
    and S is the numnber of EEG samples
 
    fs : integer.Input describing the sampling rate of EEG data in units of samples per second

    Returns
    -------
    eeg_epochs_fft : numpy array of floats of dimension T x C x S where T is the number of event TRIALS, C is the number of channels
    and S is the numnber of FFT points computed from EEG samples. eeg_epochs_fft contains the complex number spectrum of the eeg data
        
    fft_frequencies : numpuy array of float of size (n,) where n is the number of frequency number from 0 (DC) up to the nyquest rate.
        

    """
    
    # Take FFT of signal
    eeg_epochs_fft=np.fft.rfft(eeg_epochs)
    #Compute FFT Magnitude from Complex values
    eeg_epochs_fft_magnitude=np.absolute(eeg_epochs_fft-eeg_epochs_fft)
    #Compute Frequencies
    fft_frequencies=np.arange(0,fs/2,(fs/2)/eeg_epochs_fft_magnitude.shape[2])
    
    return eeg_epochs_fft,fft_frequencies 

def plot_power_spectrum(eeg_epochs_fft,fft_frequencies,is_trial_15Hz,channels,channels_to_plot,subject=1):
    """
    Function to plot the power spectrum of the eeg epochs based on the selected channels to plot
    passed as a list of strings.

    Parameters
    ----------
    eeg_epochs_fft : numpy array of floats of dimension T x C x S where T is the number of event TRIALS, C is the number of channels
    and S is the numnber of FFT points computed from EEG samples. eeg_epochs_fft contains the complex number spectrum of the eeg data
        
    fft_frequencies : numpuy array of float of size (n,) where n is the number of frequency number from 0 (DC) up to the nyquest rate.
        
    is_trial_15Hz : Boolean array of size T what is TRUE whenever the event was 15Hz SSVEP 
        
    channels : array of string containing the list of channels
       
    channels_to_plot : tuple of string whose elements are part of the elements of the "channels" array
        DESCRIPTION.
    subject : Integer, optional
        The default is 1.

    Returns
    -------
    eeg_epochs_fft_db_12hz : Numpy array of floats of size T x S where T is the number of trials and S is the number of  FFT points. eeg_epochs_fft_db_12hz
        contains the power spectrum of data across channels after data across trials was averaged and normalized to the highest value
        
    eeg_epochs_fft_db_15hz : Numpy array of floats of size T x S where T is the number of trials and S is the number of  FFT points. eeg_epochs_fft_db_12hz
        contains the power spectrum of data across channels after data across trials was averaged and normalized to the highest value

    """
    
    #Find the 12Hz trials
    is_trial_12Hz=is_trial_15Hz==False
    #separate 12Hz and 15Hz epochs
    eeg_epochs_fft_12Hz=eeg_epochs_fft[is_trial_12Hz]
    eeg_epochs_fft_15Hz=eeg_epochs_fft[is_trial_15Hz]
    
    #Compute FFT Magnitude from Complex values for 12Hz
    eeg_epochs_fft_magnitude_12hz=np.absolute(eeg_epochs_fft_12Hz)
    eeg_epochs_fft_magnitude_15hz=np.absolute(eeg_epochs_fft_15Hz)
    
    #Compute the power
    #Generate power array
    power_array=np.zeros(eeg_epochs_fft_magnitude_12hz.shape)
    power_array=2 #Array of dimension m,n,l with value=2
    #Compute the power by squaring each element
    eeg_epochs_fft_power_12hz=np.power(eeg_epochs_fft_magnitude_12hz,power_array)
    eeg_epochs_fft_power_15hz=np.power(eeg_epochs_fft_magnitude_15hz,power_array)
    #Compute the mean
    eeg_epochs_fft_mean_12hz=np.mean(eeg_epochs_fft_power_12hz, axis=0)
    eeg_epochs_fft_mean_15hz=np.mean(eeg_epochs_fft_power_15hz, axis=0)
    #Normalize to the highest power. Use array broadcasting to handle dimensions mismatch
    eeg_epochs_fft_normalized_12hz=eeg_epochs_fft_mean_12hz/np.max(eeg_epochs_fft_mean_12hz,axis=1)[:,np.newaxis]
    eeg_epochs_fft_normalized_15hz=eeg_epochs_fft_mean_12hz/np.max(eeg_epochs_fft_mean_15hz,axis=1)[:,np.newaxis]
    
    #Compute the FFT power in dB
    eeg_epochs_fft_db_12hz= np.log10(eeg_epochs_fft_normalized_12hz)
    eeg_epochs_fft_db_15hz= np.log10(eeg_epochs_fft_normalized_15hz)
    
    #is_channel_to_plot=channels==any(channels_to_plot)
    
    #Plot the spectrum
    plot_count=len(channels_to_plot)
    fig, axs = plt.subplots( plot_count,sharex=True)
    
    
    for channel_index, channel_name in enumerate(channels_to_plot):
        
       is_channel_to_plot=channels==channel_name
       axs[channel_index].plot(fft_frequencies,np.squeeze(eeg_epochs_fft_db_12hz[is_channel_to_plot]),label='12Hz')
       axs[channel_index].axvline(x=12,linewidth=1, color='b')
       axs[channel_index].plot(fft_frequencies,np.squeeze(eeg_epochs_fft_db_15hz[is_channel_to_plot]),label='15Hz')
       axs[channel_index].axvline(x=15,linewidth=1,  color="orange")
       axs[channel_index].set_ylabel('Power (dB)')
       axs[channel_index].set_xlabel('Frequency (Hz)')
       axs[channel_index].set_title(f'Channel {channel_name} frequency content\n for SSVEP S{subject}')
       axs[channel_index].legend()
       axs[channel_index].grid()
    plt.tight_layout()
    return eeg_epochs_fft_db_12hz,eeg_epochs_fft_db_15hz   
        
        
    
    
    
