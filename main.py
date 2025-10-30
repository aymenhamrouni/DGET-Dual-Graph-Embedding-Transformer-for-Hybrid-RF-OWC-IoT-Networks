# -*- coding: utf-8 -*-
"""
@author: Aymen Hamrouni
"""

import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from docplex.mp.relaxer import Relaxer
import numpy as np
import data_generation
from utils import (
    run_APs_simulation,run_simulation,plot_simulation_results,plot_network_parameters,run_Nd_simulation
)
import numpy as np



matplotlib.use('TkAgg') 
plt.rcParams.update({'font.size': 15})  # Set global font size
# # Show the plot
# plt.rcParams.update({
#     "text.usetex": True,       # Use LaTeX for text rendering
#     "font.family": "serif",    # Use serif font (like LaTeX default)
#     "pgf.rcfonts": False,      # Don't override fonts
#     "pgf.texsystem": "pdflatex",  # Use pdflatex, xelatex, or lualatex
#     "pgf.preamble": "\n".join([
#         r"\usepackage[utf8]{inputenc}",
#         r"\usepackage[T1]{fontenc}",
#         r"\usepackage{amsmath}",
#     ]),
# })
# Define dictionaries to store energy, AoI, and other metrics
EnergyPerBit_Hybrid=defaultdict(lambda: 0)
EnergyPerBit_RF=defaultdict(lambda: 0)
EnergyPerBit_WOC=defaultdict(lambda: 0)

TotalExchangedPackagesRF =defaultdict(lambda: 0)       
TotalExchangedPackagesHybrid =defaultdict(lambda: 0)    
TotalExchangedPackagesWOC =defaultdict(lambda: 0)    

PAoIRF=defaultdict(lambda: 0)       
PAoIHybrid=defaultdict(lambda: 0)    
PAoIWOC=defaultdict(lambda: 0)       

MAoIRF=defaultdict(lambda: 0)       
MAoIHybrid=defaultdict(lambda: 0)     
MAoIWOC=defaultdict(lambda: 0)     


PAoIRF_C=defaultdict(lambda: 0)       
PAoIHybrid_C=defaultdict(lambda: 0) 
PAoIWOC_C=defaultdict(lambda: 0)       


MAoIRF_C=defaultdict(lambda: 0)       
MAoIHybrid_C=defaultdict(lambda: 0)     
MAoIWOC_C=defaultdict(lambda: 0)     


HybridSwitching=defaultdict(lambda: 0)
HybridEnergy=defaultdict(lambda: 0)


objective_type='max_messages' # can be 'max_messages', 'min_energy', 'max_capacity', 'min_delay', 'min_switching', 'max_messages-w.r.t.energy', 'multi-objective-max-messages and min-energy'


# ------------------------ #
#        PARAMETERS System #
# ------------------------ #
# System parameters (network setup)
Time_unit=5e-5 # 1 unit is 50 microseconds
Time=10 # 10*50 microseconds = 500 microseconds
N_d = 5  # Number of IoT nodes
N_APs =2  # Number of Access Points (APs)
N = N_d + N_APs  # Total number of devices

S_p=128 #size of a packet in bits
snr_min_woc = 10  # Minimum SNR for good communication in WOC (in dB)
snr_min_rf = 15.0  # Minimum SNR for good communication in RF sub-bands (in dB)
N_montecarlo=1 # Number of Monte Carlo simulations



start=1*S_p # 1 packet
end=2000*S_p # 4000 packets
step=200*S_p #200 packets
           



VIZ=False # print simulation figures or not
Verbose=False # print simulation details or not



for _NC in range(0,N_montecarlo):
    if Verbose:
            print(f"\nRunning Monte Carlo simulation {_NC + 1}/{N_montecarlo}")
    
    # Create network characteristics object
    network = data_generation.NetworkCharacteristics(numAPs=N_APs, numDevices=N_d, num_time_slots=Time)
    
    # Generate network parameters with MinD, MaxD, MinE, MaxE (Distance and Energy)
    (distance_matrix, snr_db_matrix_WOC, capacity_matrix_WOC,
     snr_db_matrix_RF, capacity_matrix_RF, snr_matrix, capacity_matrix,
     SendEnergy, RecieveEnergy, EnergyTotal) = network.generate_network_parameters(minD=5, maxD=20, minE=0.05, maxE=0.1)
    
    if VIZ==True:
        plot_network_parameters(
            snr_db_matrix_WOC, capacity_matrix_WOC,
            snr_db_matrix_RF, capacity_matrix_RF,
            capacity_matrix
        )

    # Generate messages with MinSize, MaxSize, MinAppSize, MaxAppSize
    T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp = network.generate_messages(minSize=1, maxSize=3, appSize=2, data_gen_prob=0.7 )
    
    
  
    for size in range(int(start), int(end) + 1, int(step)):
        MsgSize, MsgNumber, maxN = network.generate_size(msgQueues, sizemin=size, sizemax=size, S_p=S_p)
        if Verbose:
            print(f'Current Size Generation: {size}, Max Message Length in Sp: {maxN}')
        # Run Hybrid simulation
        results_hybrid = run_simulation(
            M_WOC=0, M=2, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
            BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
            snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC,
            snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
            capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=Time_unit,
            SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
            Verbose=Verbose, VIZ=VIZ,objective_type=objective_type, Timeout=60, msgApp=msgApp, N_APs=N_APs
        )
         
        # Run RF-only simulation
        results_rf = run_simulation(
            M_WOC=0, M=1, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
            BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
            snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC,
            snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
            capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=Time_unit,
            SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
            Verbose=Verbose, VIZ=VIZ, objective_type=objective_type, Timeout=60, msgApp=msgApp, N_APs=N_APs
        )
        
        # Run WOC-only simulation
        results_woc = run_simulation(
            M_WOC=1, M=2, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
            BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
            snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC,
            snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
            capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=Time_unit,
            SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
            Verbose=Verbose, VIZ=VIZ,objective_type=objective_type, Timeout=60, msgApp=msgApp,N_APs=N_APs
        )
        
       # Store results in appropriate dictionaries
        if results_hybrid:
            EnergyPerBit_Hybrid[int(size/S_p)] += results_hybrid['energy']/results_hybrid['total_bits_exchanged']
            TotalExchangedPackagesHybrid[int(size/S_p)] += results_hybrid['total_bits_exchanged']/S_p
            MAoIHybrid[int(size/S_p)] += results_hybrid['mean_aoi_all']
            PAoIHybrid[int(size/S_p)] += results_hybrid['peak_aoi_all']
            MAoIHybrid_C[int(size/S_p)] += results_hybrid['mean_aoi_communicating']
            PAoIHybrid_C[int(size/S_p)] += results_hybrid['peak_aoi_communicating']
            HybridSwitching[int(size/S_p)] += results_hybrid['switching']
            HybridEnergy[int(size/S_p)] += results_hybrid['energy']
        if results_rf:
            EnergyPerBit_RF[int(size/S_p)] += results_rf['energy']/results_rf['total_bits_exchanged']
            TotalExchangedPackagesRF[int(size/S_p)] += results_rf['total_bits_exchanged']/S_p
            MAoIRF[int(size/S_p)] += results_rf['mean_aoi_all']
            PAoIRF[int(size/S_p)] += results_rf['peak_aoi_all']
            MAoIRF_C[int(size/S_p)] += results_rf['mean_aoi_communicating']
            PAoIRF_C[int(size/S_p)] += results_rf['peak_aoi_communicating']
        
        if results_woc:
            EnergyPerBit_WOC[int(size/S_p)] += results_woc['energy']/results_woc['total_bits_exchanged']
            TotalExchangedPackagesWOC[int(size/S_p)] += results_woc['total_bits_exchanged']/S_p
            MAoIWOC[int(size/S_p)] += results_woc['mean_aoi_all']
            PAoIWOC[int(size/S_p)] += results_woc['peak_aoi_all']
            MAoIWOC_C[int(size/S_p)] += results_woc['mean_aoi_communicating']
            PAoIWOC_C[int(size/S_p)] += results_woc['peak_aoi_communicating']




        
    plot_simulation_results(
        EnergyPerBit_RF, EnergyPerBit_Hybrid, EnergyPerBit_WOC,
        TotalExchangedPackagesRF, TotalExchangedPackagesHybrid, TotalExchangedPackagesWOC,
        MAoIRF, MAoIHybrid, MAoIWOC, MAoIRF_C, MAoIHybrid_C, MAoIWOC_C,
        PAoIRF, PAoIHybrid, PAoIWOC, PAoIRF_C, PAoIHybrid_C, PAoIWOC_C, HybridSwitching, HybridEnergy,
        N_montecarlo,Type='Energy/Throughput'
    )

# Run network size simulation
# Network size simulation parameters

print(f"\nRunning network size simulation with varying N_APs and fixed N_d")
run_APs_simulation(
    N_APs_range=range(2, 3),
    N_d=5,
    fixed_size=200*S_p,
    minD=5,
    maxD=20,
    minE=0.05,
    maxE=0.1,
    Time=Time,
    N_montecarlo=N_montecarlo,
    Verbose=Verbose,
    VIZ=VIZ,
    objective_type='max_messages',
    Timeout=60,
    snr_min_rf=snr_min_rf,
    snr_min_woc=snr_min_woc,  
    S_p=S_p
)

print(f"\nRunning network size simulation with varying N_d and fixed N_APs")
run_Nd_simulation(
    N_d_range=range(5, 6),
    N_APs=3,
    fixed_size=200*S_p,
    minD=5,
    maxD=20,
    minE=0.05,
    maxE=0.1,
    Time=Time,
    N_montecarlo=N_montecarlo,
    Verbose=Verbose,
    VIZ=VIZ,
    objective_type='max_messages',
    Timeout=60,
    snr_min_rf=snr_min_rf,
    snr_min_woc=snr_min_woc,
    S_p=S_p
)


