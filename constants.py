"""
System Constants and Parameters for Hybrid RF-WOC Network Optimization

This module contains all the physical layer constants, system parameters, and configuration
values used throughout the simulation. These include:

1. Physical Layer Constants:
   - Fundamental constants (q, k_B)
   - Environmental parameters (Temperature)
   - Hardware specifications (R_L)

2. WOC (Wireless Optical Communication) Parameters:
   - Illumination and optical characteristics
   - Receiver specifications
   - Channel parameters

3. RF (Radio Frequency) Parameters:
   - Transmission parameters
   - Path loss model constants
   - Antenna characteristics

4. System Configuration:
   - Time units and simulation parameters
   - Network topology settings
   - Performance thresholds

Author: Aymen Hamrouni
Date: 2024
"""

import numpy as np
markers = (
    ".",   # Point marker
    ",",   # Pixel marker
    "o",   # Circle marker
    "v",   # Triangle down marker
    "^",   # Triangle up marker
    "<",   # Triangle left marker
    ">",   # Triangle right marker
    "1",   # Tri-down marker
    "2",   # Tri-up marker
    "3",   # Tri-left marker
    "4",   # Tri-right marker
    "s",   # Square marker
    "p",   # Pentagon marker
    "*",   # Star marker
    "h",   # Hexagon1 marker
    "H",   # Hexagon2 marker
    "+",   # Plus marker
    "x",   # X marker
    "D",   # Diamond marker
    "d",   # Thin diamond marker
    "|",   # Vertical line marker
    "_",   # Horizontal line marker
)
# Physical layer constants
q = 1.602e-19  # Charge of an electron (Coulombs)
k_B = 1.381e-23  # Boltzmann constant (J/K)
Temperature = 290  # Temperature in Kelvin (room temperature)
R_L = 50  # Load resistance in Ohms

# WOC parameters
IlluminationCoefficient = 0.3  # Illumination efficiency factor
f_WOC = 100e9  # WOC carrier frequency (Hz)
B_WOC = 1e9  # WOC bandwidth (Hz)
P_tx_WOC = 5  # WOC transmit power (W)
T_optical = 0.99  # Optical transmittance
R = 0.99  # Reflectance
R_photo = 0.6  # Photodetector responsivity (A/W)
lamda = 3e8 / f_WOC  # Wavelength (m)
u = 1  # Lambertian order
theta = phi = 60  # Half-power semiangle (degrees)
A_rec = 0.0025  # Receiver area (m²)
P_N = 10**(-21)  # Noise power (W)
V_bias = 3  # Photodetector bias voltage (V)

# RF parameters
f_RF = 2.4e9  # RF carrier frequency (Hz)
lambda_RF = 3e8 / f_RF  # RF wavelength (m)
B_RF = 2e6  # RF bandwidth (Hz)
P_tx_RF = 0.05  # RF transmit power (W)
attenuation_coefficient_RF = 3  # Path loss exponent
d0 = 1  # Reference distance (m)
PL0 = 20 * np.log10(4 * np.pi * d0 / lambda_RF)  # Reference path loss (dB)
sigma_shadowing = 10  # Shadowing standard deviation (dB)
G_tx_RF = 5  # RF transmit antenna gain (dBi)
G_rx_RF = 5  # RF receive antenna gain (dBi)

# System parameters
transferTime = 1  # Time for transfer of one packet (time units)