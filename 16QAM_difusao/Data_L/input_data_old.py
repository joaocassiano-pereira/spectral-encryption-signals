# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:44:43 2021

@author: pc
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:27:38 2020

@author: pc
"""

from scipy import constants
import numpy as np

'''|====|====|====|Constantes|====|====|====|'''

c = constants.c

pi = constants.pi

'''|====|====|====|Parametros Gerais|====|====|====|'''

#Symbol rate.
Rsymbol= 28e9

#Number of symbols.
Nsymbols = 4*4096
#Nsymbols= 4*4096

#Number of samples per symbol.
spsymbol= 2

#Number of initial and final symbols with null amplitudes.
N0= 0

#Number of samples per spectral encoding slice.
spslice=1

'''Factor of a raised cosine filter. The signal is
passed through this filter to limit its bandwidth. The signal may also be
passed through an equalizer that excatly compensates the attenuation imposed
by the raised cosine filter.'''
roll_off= 0.02 

''''Generates (1) or reads (0) prss' (pseudo-random symbol sequence) 
from a file specified in prbs_filename1 and/or prbs_filename2'''
prss= 1

#Considers a real (0) or a complex (1) signal
Q= 0

#Number of symbols in the I-axis
Symbols_I= 4

#Number of symbols in the Q-axis
Symbols_Q= 4

#Scrambles (1) or not (0) the signal
scramble= 0

#Angle-encodes (1) or not (0) the signal 
angle_encoding= 1

#Enables (1) or disables (0) the random delay encoding
enable_delay= 0

#Minimum delay encoding (in units of symbol periods) 
min_delay= 0.5

#Maximum delay encoding (in units of symbol periods)
max_delay= 20

#Pulse profile type
pulse_type = 'retangular'

'''Turns on (1) and off (0) a random error of phase_min
                      % around the exact slice decoding angle '''
random_on= 0

#Random error around the exact slice decoding angle (in degrees)
phase_min_deg= 30

'''The ratio between the maximum amplitude provided by 
                      % the Digital to Analog Converter (DAC), mp, and the standard deviation
                      % of the DAC input signal, sigma. Thus:
                      %  mp= loading_factor*sigma.'''
loading_factor= 3

#Maximum Digital to Analog Converter (DAC) amplitude
#dac_max= 5 

#DAC number of output levels
dac_levels= 55


#Amplifies/ attenuates a signal before bit demappinf is performed. 
demapper_gain= 1

'''noise'''
#Enables (1) or disables (0) the noise in the Rx
#noise_on= 0
#noise standard deviation
sigma_noise= 0;
#noise average
noise_avg = 0;  
'''|====|====|====|Parametros Calculados|====|====|====|'''
#Number of samples
nsamples= Nsymbols*spsymbol

#Symbol period 
Tsymbol= 1/Rsymbol

#Sampling period
Ts= Tsymbol/spsymbol

#Sampling frequency. This is also the maximum frequency in all spectra. 
Fs= 1/Ts

#Signal Bandwidth
Bsignal= Rsymbol*(1+roll_off)/2

#Banda do Filtro
Filterbandwidth = Rsymbol 

'''Benc= Bsignal;                    % Bandwidth to be encoded'''
#Bandwidth to be encoded
Benc= Rsymbol*(1+roll_off)/2

#Number of encoded slices
n_slice = int(np.ceil(((nsamples)/Fs)*Benc/spslice))

#Slice bandwidth
Bslice = Benc/n_slice

#Number of spectral components that will be scrambled
n_scramble_positive = np.ceil((nsamples-1)*Benc/Fs)
n_scramble_negative = nsamples-n_scramble_positive+1;

#Number of spectral components that will be scrambled
n_scramble = int(2*n_scramble_positive)

#Number of spectral components within the signal bandwidth
n_signal = np.ceil((nsamples)*Bsignal/Fs)

#Random error around the exact slice decoding angle (in radians)
phase_min= phase_min_deg*pi/180

#number of bits per symbol in I-axis
nbpsy_I = int(np.ceil(np.log2(Symbols_I)))

#number of bits per symbol in Q-axis
nbpsy_Q = int(np.ceil(np.log2(Symbols_Q)))

#number of bits in I-axis
nbits_I = (Nsymbols*nbpsy_I)

#number of bits in Q-axis
nbits_Q = (Nsymbols*nbpsy_Q)

#Time and Frequency parameters
time = [Ts*i for i in range(0,nsamples)];  #numero de pontos no eixo x
frequency= [Fs*i/(nsamples-1) for i in range(0,nsamples)];

(
nsamples, 
Tsymbol, 
Ts, 
Fs, 
Bsignal, 
Filterbandwidth, 
Benc, 
n_slice, 
Bslice, 
n_scramble_positive, 
n_scramble_negative, 
n_scramble, n_signal, 
phase_min, 
nbpsy_I, 
nbpsy_Q, 
nbits_I, 
nbits_Q, 
time, 
frequency
)


'''|====|====|====|Miscelanea|====|====|====|'''

if Symbols_I == 2 and Symbols_Q == 1:
    modu = "BPSK"
if Symbols_I == 2 and Symbols_Q == 2:
    modu = "QPSK"
if Symbols_I == 4 and Symbols_Q == 4:
    modu = "16-QAM"


'''|====|====|====|Miscelanea|====|====|====|'''

melSNRDBspeBPSK = [15.55451,
       14.60033,
       13.83605,
       13.23817,
       12.36771,
       11.47543,
       9.77438,
       9.32553,
       8.96728,
       8.74791]

melBERspeBPSK = [1.02293E-9,
        3.92543E-8,
        4.3681E-7,
        2.20549E-6,
        1.63911E-5,
        8.92113E-5,
        0.00103,
        0.00172,
        0.00249,
        0.00309]

melSNRDBspeQPSK = [18.27141,
                17.64411,
                17.09814,
                15.83805,
                15.31826,
                13.73592,
                12.76388,
                12.27813,
                12.15556,
                11.69801]

melBERspeQPSK = [3.41544E-9,
                3.49818E-8,
                2.06558E-7,
                5.95784E-6,
                1.85567E-5,
                2.93229E-4,
                0.00106,
                0.00183,
                0.00208,
                0.00328]

melSNRDBspe16QAM = [25.24422,
                    24.65025,
                    23.97596,
                    22.80692,
                    22.23073,
                    21.04567,
                    19.46001,
                    18.97443,
                    18.68742,
                    18.29366]

melBERspe16QAM = [2.73919E-9,
                2.47854E-8,
                2.17205E-7,
                4.68839E-6,
                1.62995E-5,
                1.35493E-4,
                0.00111,
                0.00186,
                0.00246,
                0.00351]

melSNRDBsencBPSK = [15.2676,
                    14.61949,
                    13.93019,
                    13.29624,
                    12.26616,
                    11.1813,
                    9.61602,
                    9.15889,
                    8.85273,
                    8.72018]

melBERsencBPSK = [3.32844E-9,
                3.67539E-8,
                3.31747E-7,
                1.90203E-6,
                2.02198E-5,
                1.45619E-4,
                0.00124,
                0.00205,
                0.00279,
                0.00318]

melSNRDBsencQPSK = [18.40198,
                    17.54188,
                    16.94992,
                    15.90845,
                    15.37303,
                    13.97296,
                    12.7088,
                    12.22235,
                    11.87137,
                    11.53896]

melBERsencQPSK = [2.01694E-9,
                4.9601E-8,
                3.22631E-7,
                5.05603E-6,
                1.65627E-5,
                2.05505E-4,
                0.00113,
                0.00194,
                0.00277,
                0.0038]

melSNRDBsenc16QAM = [25.43836,
                    24.71069,
                    23.93633,
                    23.34167,
                    22.45457,
                    21.17295,
                    19.48994,
                    19.02069,
                    18.50735,
                    18.31792]

melBERsenc16QAM = [1.24831E-9,
                    2.00709E-8,
                    2.44337E-7,
                    1.26885E-6,
                    1.02343E-5,
                    1.10694E-4,
                    0.00107,
                    0.00177,
                    0.0029,
                    0.00344]
