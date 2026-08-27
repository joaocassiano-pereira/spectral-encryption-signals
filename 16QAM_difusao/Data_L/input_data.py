# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:27:38 2020

@author: pc
"""

import configparser
import os
from pathlib import Path

from scipy import constants
import numpy as np


'''|====|====|====|Constants|====|====|====|'''

c = constants.c

pi = constants.pi

'''|===|Config Parser|==|'''
config = configparser.ConfigParser()
config.optionxform = str

# config['SETTINGS'] = {'Rsymbol': '28e9',
#                       'Nsymbols': '16384',
#                       'spsymbol': '2',
#                       'N0': '0',
#                       'spslice': '1',
#                       'roll_off':'0.02',
#                       'prss': '1',
#                       'Q':'0',
#                       'Symbols_I': '2',
#                       'Symbols_Q': '1',
#                       'scramble': '0',
#                       'angle_encoding': '1',
#                       'enable_delay': '0',
#                       'min_delay': '0.5',
#                       'max_delay': '20',
#                       'pulse_type' : 'retangular',
#                       'random_on': '0',
#                       'phase_min_deg': '30',
#                       'loading_factor': '3',
#                       'dac_levels': '55',
#                       'demapper_gain': '1',
#                       'sigma_noise': '0.5',
#                       'noise_avg': '0'
#                       }
# with open('example.ini', 'w') as configfile:
#     config.write(configfile)
script_dir = Path(__file__).parent
config.read(str(script_dir)+"\settings.ini")

'''|====|====|====|Input Parameters|====|====|====|'''

# Symbol rate (Baud).
Rsymbol = config.getfloat('SETTINGS', 'Rsymbol')

# Number of symbols.
Nsymbols = config.getint('SETTINGS', 'Nsymbols')

# Number of samples per symbol.
spsymbol = config.getint('SETTINGS', 'spsymbol')

# Number of initial and final symbols with null amplitudes.
N0 = config.getfloat('SETTINGS', 'N0')

# Number of samples per spectral encoding slice.
spslice = config.getfloat('SETTINGS', 'spslice')

'''Factor of a raised cosine filter. The signal is
passed through this filter to limit its bandwidth. The signal may also be
passed through an equalizer that excatly compensates the attenuation imposed
by the raised cosine filter.'''
roll_off = config.getfloat('SETTINGS', 'roll_off')

''''Generates (True) or reads (False) prss' (pseudo-random symbol sequence)
from a file specified in prbs_filename1 and/or prbs_filename2'''
prss = config.getboolean('SETTINGS', 'prss')

'''Enables (True) or disables (False) gray coding'''
gray = config.getboolean('SETTINGS', 'gray')

# Considers a real (False) or a complex (True) signal
Q = config.getboolean('SETTINGS', 'Q')

# Number of symbols in the I-axis
Symbols_I = config.getint('SETTINGS', 'Symbols_I')

# Number of symbols in the Q-axis
Symbols_Q = config.getint('SETTINGS', 'Symbols_Q')

# Scrambles (True) or not (False) the signal
scramble = config.getboolean('SETTINGS', 'scramble')

# Angle-encodes (True) or not (False) the signal
angle_encoding = config.getboolean('SETTINGS', 'angle_encoding')

# Enables (True) or disables (False) the random delay encoding
enable_delay = config.getboolean('SETTINGS', 'enable_delay')

# Minimum delay encoding (in units of symbol periods)
min_delay = config.getfloat('SETTINGS', 'min_delay')

# Maximum delay encoding (in units of symbol periods)
max_delay = config.getfloat('SETTINGS', 'max_delay')

# Pulse profile type
pulse_type = config.get('SETTINGS', 'pulse_type')

'''Turns on (True) and off (False) a random error of phase_min
                      % around the exact slice decoding angle '''
random_on = config.getboolean('SETTINGS', 'random_on')

# Random error around the exact slice decoding angle (in degrees)
phase_min_deg = config.getfloat('SETTINGS', 'phase_min_deg')

'''The ratio between the maximum amplitude provided by
                      % the Digital to Analog Converter (DAC), mp, and the standard deviation
                      % of the DAC input signal, sigma. Thus:
                      %  mp= loading_factor*sigma.'''
loading_factor = config.getfloat('SETTINGS', 'loading_factor')

# Maximum Digital to Analog Converter (DAC) amplitude
# dac_max= 5

# DAC number of output levels
dac_levels = config.getfloat('SETTINGS', 'dac_levels')

# Amplifies/ attenuates a signal before bit demappinf is performed.
demapper_gain = config.getfloat('SETTINGS', 'demapper_gain')

'''noise'''
# Enables (True) or disables (False) the noise in the Rx
# noise_on= 0
# noise standard deviation
sigma_noise = config.getfloat('SETTINGS', 'sigma_noise')
# noise average
noise_avg = config.getfloat('SETTINGS', 'noise_avg')

'''|====|====|====|Evaluated Parameters|====|====|====|'''
# Number of samples
nsamples = Nsymbols*spsymbol

# Symbol period
Tsymbol = 1/Rsymbol

# Sampling period
Ts = Tsymbol/spsymbol

# Sampling frequency. This is also the maximum frequency in all spectra.
Fs = 1/Ts

# Signal Bandwidth
Bsignal = Rsymbol*(1+roll_off)/2

# Filter Bandwidth
Filterbandwidth = Rsymbol

'''Benc= Bsignal;                    % Bandwidth to be encoded'''
# Bandwidth to be encoded
Benc = Rsymbol*(1+roll_off)/2

# Number of encoded slices
n_slice = int(np.ceil(((nsamples)/Fs)*Benc/spslice))

# Slice bandwidth
Bslice = Benc/n_slice

# Number of spectral components that will be scrambled
n_scramble_positive = np.ceil((nsamples-1)*Benc/Fs)
n_scramble_negative = nsamples-n_scramble_positive+1

# Number of spectral components that will be scrambled
n_scramble = int(2*(n_scramble_positive))

# Number of spectral components within the signal bandwidth
n_signal = np.ceil((nsamples)*Bsignal/Fs)

# Random error around the exact slice decoding angle (in radians)
phase_min = phase_min_deg*pi/180

# number of bits per symbol in I-axis
nbpsy_I = int(np.ceil(np.log2(Symbols_I)))

# number of bits per symbol in Q-axis
nbpsy_Q = int(np.ceil(np.log2(Symbols_Q)))

# number of bits in I-axis
nbits_I = (Nsymbols*nbpsy_I)

# number of bits in Q-axis
nbits_Q = (Nsymbols*nbpsy_Q)

# Time and Frequency parameters
time = [Ts*i for i in range(0, nsamples)]  # intervals of time
frequency = [Fs*i/(nsamples-1) for i in range(0, nsamples)]

'''|====|====|====|Misc|====|====|====|'''

if Symbols_I == 2 and Symbols_Q == 1:
    modu = "BPSK"
if Symbols_I == 2 and Symbols_Q == 2:
    modu = "QPSK"
if Symbols_I == 4 and Symbols_Q == 4:
    modu = "16-QAM"

'''Test Data'''

# Mel_data_fig4 = np.load('Mel_data_fig4.npz',allow_pickle=True)['data'].item()
