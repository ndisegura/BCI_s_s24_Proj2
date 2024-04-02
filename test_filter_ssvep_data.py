# -*- coding: utf-8 -*-
"""
Andres Segura & Tynan Gacy
BME 6770: BCI's Lab 04
Dr. David Jangraw
3/11/2024

Test scrip for the implementation of the filter_ssvep_data module. This scrip loads the data from the data dictionary, 
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


#%% Cell1 Load the Data

data=import_ssvep_data.load_ssvep_data(subject,data_directory)

#%% Cell2 Design a Filter

#Filter out the 15Hz signals
low_cutoff=13
high_cutoff=17
filter_type='hann'
filter_order=1001
fs=data['fs']
filter_coefficients_band_pass_12Hz=filter_ssvep_data.make_bandpass_filter(low_cutoff,high_cutoff,filter_type,filter_order,fs)

#Filter out the 12Hz signals
low_cutoff=10
high_cutoff=14
filter_type='hann'
filter_order=1001
fs=data['fs']
filter_coefficients_band_pass_15Hz=filter_ssvep_data.make_bandpass_filter(low_cutoff,high_cutoff,filter_type,filter_order,fs)

"""
A) How much will 12Hz oscillations be attenuated by the 15Hz filter? How much will 15Hz 
oscillations be attenuated by the 12Hz filter?

According to the frequency response plot. Each frequency will be attenuated by approximately  -27dBc

B) Experiment with higher and lower order filters. Describe how changing the order changes 
the frequency and impulse response of the filter

Increasing the order of the filter improves the attenuation of the adjacent signals (e.g. 15Hz for the 12Hz Bandpass filter),
but also incrases the lenght of the impulse response. Similarly, reducing the order of the filter decrases the attenuation of 
adjacent frequencies, but it also reduces the length of the impulse response.

"""
#%% Cell 3 Filter the EEG signals

#Filter 12Hz signals
filtered_data_12Hz=filter_ssvep_data.filter_data(data,filter_coefficients_band_pass_12Hz)

#Filtered 15Hz signals
filtered_data_15Hz=filter_ssvep_data.filter_data(data,filter_coefficients_band_pass_15Hz)


#%% Cell 4 Calculate the Envelope

# Get envelope for 12Hz
envelope_12Hz=filter_ssvep_data.get_envelope(data, filtered_data_12Hz[:,:],'Oz',12)
#envelope_12Hz=filter_ssvep_data.get_envelope(data, filtered_data_12Hz[:,145000:164000],'Oz',12)

# Get envelope for 15Hz
envelope_15Hz=filter_ssvep_data.get_envelope(data, filtered_data_15Hz[:,:],'Oz',15)
#envelope_15Hz=filter_ssvep_data.get_envelope(data, filtered_data_15Hz[:,145000:164000],'Oz',15)

#%% Cell 5 Plot amplitudes

# Plot amplitudes for Oz

filter_ssvep_data.plot_ssvep_amplitudes(data,envelope_12Hz,envelope_15Hz,'Oz',12,15,subject)

"""
What do the two envelopes do when the stimulation frequency changes? 
 - The voltage for the 12Hz envelope stays relatively constant but the 15Hz 
envelope has spikes in signal when the 15Hz frequency is the focus.

How large and consistent are those changes? 
 - The spikes are dramatic and occur immediately after the end of the flash frequency.
They are all large, but the peak voltage is not consistent.

Are the brain signals responding to the events in the way you’d expect? 
 - These are SSVEPs, brain signals that synchronize in frequency to the flashing
 frequency that is being focused on. The 15Hz amplitude spike when the 15Hz
 flash frequency is occuring makes sense. We did expect to see the same for 12Hz
 which was not observed, however.

Check some other electrodes – which electrodes respond in the same way and why?
 - O1, O2 are the most similar, but P3, Pz, and P4 also had strong responses. 
They are all close to the visual cortex of the occipital lobe and would contribute
to SSVEPs.
"""



#%% Cell 6 Examine Spectra

filter_ssvep_data.plot_filtered_spectra(data,filtered_data_15Hz,envelope_15Hz,{'Fz','Oz'})

"""
describe how the spectra change at each stage and why. Changes you should address include (but are not limited to) the following:
 - In the raw epochs, there is a lot of noise. This noise is reduced when the 
 data is filtered, isolating the SSVEP signal. A significant source of noise is 
 an artificial artifact at intervals of 50Hz, which is removed. Other major sources
 of noise, like DC offset occuring near 0Hz are also removed. It also removes 
 the harmonic frequencies of noise which eliminates many higher frequencies, 
 so the curve becomes more logarhythmic in shape and less flat.  
    
 - While the filtered data show the presence of frequencies in the sampled data, 
 envelopes then show the change in overall magnitude for the signal over the same
 time. This tells us that the amplitude for 15Hz starts high and decreases through
 the epoch. 
    
 - Because 12Hz is within the frequency for alpha waves, the baseline level of 
 activity for it is higher than 15Hz. On channels that are not observing brain 
 regions with variable activity within the recording, it will have higher power.
 
 - Because the envelope is measuring the amplitude of the signal over time, there 
 isn't a peak at 15Hz. The peak observed in the envelope at 0 Hz represents the 
 overall amplitude modulation of the signal, rather than the presence of a specific 
 frequency component. 
 
"""



