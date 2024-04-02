# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:25:50 2024

@author: asegura
"""

# Inport the necessry modules
import os
import sys
import matplotlib.pyplot as plt
import import_ssvep_data

#Make sure relative path work
cwd=os.getcwd()
sys.path.insert(0,f"{cwd}\course_software\SsvepData\\")

#Close previosly drawn plots
plt.close('all')

#Build data file string
data_directory='./SsvepData/'
subject=1
#data_file=f'{cwd}{data_directory}SSVEP_S{subject}.npz'

#%% Cell1 Load the Data

data_dict=import_ssvep_data.load_ssvep_data(subject,data_directory)

#%% Cell2 Plot the data

import_ssvep_data.plot_raw_data(data_dict,subject,['Fz','Oz','F4'])

#%% Cell3 Epoch the data_dict

#epoch start times
epoch_start_time = 0
epoch_end_time= 20

#epoch data

eeg_epochs,epoch_time,is_trial_15Hz=import_ssvep_data.epoch_ssvep_data(data_dict,epoch_start_time,epoch_end_time)

#%% Cell4 Compute FFT

eeg_epochs_fft,fft_frequencies=import_ssvep_data.get_frequency_spectrum(eeg_epochs,data_dict['fs'])

#%% Cell 5: Plot the Power Spectra

channels=data_dict['channels']
channels_to_plot=['Fz','Oz','F4']
spectrum_db_12Hz,spectrum_db_15Hz=import_ssvep_data.plot_power_spectrum(eeg_epochs_fft,fft_frequencies,is_trial_15Hz,channels,channels_to_plot,subject)

#%% Cell 6: 
'''
1. The signal that leads to these peaks at 12hz and 15hz for the respective 
trials is a steady state visual evoked potential (SSVEP). The steady state 
visual evoked potential originates within the occipital lobes primary visual 
cortex after visual stimuli is propagated from the temporal retina via the optic
nerve and optic chiasma. The job of the primary visual cortex is to notice 
visual information, then send it to other parts of the brain for object location
and recignition. 
2. The smaller peaks at integer multiples are caused by the harmonics of the 
fundamental SSVEP frequency.
3. The most likley cause for this peak is that the data was collected in europe
where the powerline noise occurs at 50hz. 
4. The brain signal causing extra power at 10hz is the alpha rthym. The alpha 
rthym originayes in the thalamus and is seen as a collection of alpha ruthyms
together. This may imply synchronicity between cells which increases the amplitude
of the overall alpha wave. 

'''
