# -*- coding: utf-8 -*-
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

from pylab import *
from scipy.signal import firwin, filtfilt,freqz,hilbert
import matplotlib.pyplot as plt
import numpy as np
import import_ssvep_data

