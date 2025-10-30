# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:02:18 2024

@author: Aymen Hamrouni
"""


import csv
from collections import defaultdict
from docplex.mp.relaxer import Relaxer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import data_generation
from MINLP_model import MINLP_model
from constants import markers, transferTime
import seaborn as sns
marker = itertools.cycle(markers) 
import matplotlib.ticker as ticker
import tikzplotlib
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


def plot_network_parameters(snr_db_matrix_WOC, capacity_matrix_WOC, snr_db_matrix_RF, capacity_matrix_RF, capacity_matrix):
    """
    Plot network parameters including SNR and capacity distributions.
    
    Args:
        snr_db_matrix_WOC: SNR matrix for WOC
        capacity_matrix_WOC: Capacity matrix for WOC
        snr_db_matrix_RF: SNR matrix for RF
        capacity_matrix_RF: Capacity matrix for RF
        capacity_matrix: Combined capacity matrix
    """
    plt.figure(figsize=(12, 6))
    
    # WOC SNR Histogram
    plt.subplot(1, 2, 1)
    plt.hist(snr_db_matrix_WOC.flatten(), bins=50, color='blue', alpha=0.7, label='SNR (dB)')
    plt.title('Histogram of SNR - WOC')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # WOC Capacity Histogram
    plt.subplot(1, 2, 2)
    plt.hist(capacity_matrix_WOC.flatten()/1e9, bins=50, color='green', alpha=0.7, label='Capacity (bits/s)')
    plt.title('Histogram of Channel Capacity - WOC')
    plt.xlabel('Capacity (Gbits/s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    
    # RF SNR Histogram
    plt.subplot(1, 2, 1)
    plt.hist(snr_db_matrix_RF.flatten(), bins=50, color='blue', alpha=0.7, label='SNR (dB)')
    plt.title('Histogram of SNR - RF')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # RF Capacity Histogram
    plt.subplot(1, 2, 2)
    plt.hist(capacity_matrix_RF.flatten()/1e9, bins=50, color='green', alpha=0.7, label='Capacity (bits/s)')
    plt.title('Histogram of Channel Capacity - RF')
    plt.xlabel('Capacity (Gbits/s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    # Combined Capacity Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(capacity_matrix.flatten()/1e9, bins=50, color='red', alpha=0.7, label='Capacity (bits/s)')
    plt.title('Histogram of Combined Channel Capacity - RF/WOC')
    plt.xlabel('Capacity (Gbits/s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
def plot_simulation_results(EnergyPerBit_RF, EnergyPerBit_Hybrid, EnergyPerBit_WOC,
                          TotalExchangedPackagesRF, TotalExchangedPackagesHybrid, TotalExchangedPackagesWOC,
                          MAoIRF, MAoIHybrid, MAoIWOC, MAoIRF_C, MAoIHybrid_C, MAoIWOC_C,
                          PAoIRF, PAoIHybrid, PAoIWOC, PAoIRF_C, PAoIHybrid_C, PAoIWOC_C, HybridSwitching, HybridEnergy,
                          N_montecarlo, Type='Energy/Throughput'):
    """
    Plot simulation results including energy per bit, total exchanged packets, and AoI metrics.
    """
    # Sort keys for consistent plotting
    sorted_keys = sorted(MAoIHybrid.keys())
    plt.rcParams.update({'font.size': 23})

    if Type == 'Energy/Throughput':
        sorted_keys = sorted(EnergyPerBit_RF.keys())

        # 1. Energy per bit and total exchanged packets plot
        fig, ax1 = plt.subplots(figsize=(10, 8))
        
        # Convert energy values to pJ
        EperBRF = [float(EnergyPerBit_RF[key]/1e-9) for key in sorted_keys]
        EperBHybrid = [float(EnergyPerBit_Hybrid[key]/1e-9) for key in sorted_keys]
        EperBWOC = [float(EnergyPerBit_WOC[key]/1e-9) for key in sorted_keys]
        
        # Plot energy per bit
        line1, = ax1.plot(sorted_keys, EperBRF, color='red', label='RF (Energy)',marker='*',markevery=20,markersize=15)
        line2, = ax1.plot(sorted_keys, EperBHybrid, color='green', label='RF-OWC (Energy)',marker='^',markevery=20,markersize=15)
        line3, = ax1.plot(sorted_keys, EperBWOC, color='orange', label='OWC (Energy)',marker='>',markevery=20,markersize=15)
        
        ax1.set_xlabel('Avg. Number of Packets/Device')
        ax1.set_ylabel('Energy Per Bit (nJ)')
        ax1.set_yscale('log')
        plt.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=1)
        
        # Plot total exchanged packets on secondary axis
        ax2 = ax1.twinx()
        line4, = ax2.plot(sorted_keys, [TotalExchangedPackagesRF[key] for key in sorted_keys], 
                color='red', linestyle='--', label='RF (Packets)',markevery=10,linewidth=4)
        line5, = ax2.plot(sorted_keys, [TotalExchangedPackagesHybrid[key] for key in sorted_keys], 
                color='green', linestyle='--', label='RF-OWC (Packets)',markevery=10,linewidth=4)
        line6, = ax2.plot(sorted_keys, [TotalExchangedPackagesWOC[key] for key in sorted_keys], 
                color='orange', linestyle='--', label='OWC (Packets)',markevery=10,linewidth=4)
        
        ax2.set_ylabel('Total Exchanged Packets', color='black', fontsize=20)
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Combine legends
        lines1 = [line1, line2, line3]
        labels = [line.get_label() for line in lines1]
        ax1.legend(lines1, labels, loc='upper center', fontsize=15)
        lines2= [line4, line5, line6]
        labels2= [line.get_label() for line in lines2]
        ax2.legend(lines2, labels2, loc='center right', fontsize=15)
        ax1.grid(axis='y', linestyle='dotted', alpha=0.7, color='gray', linewidth=2)
        ax2.grid(axis='y', linestyle='--', alpha=0.7, color='gray', linewidth=0.8)
        
        #tikzplotlib.save("EnergyPerbit_TTexchanged.tex")
        #plt.savefig("EnergyPerbit_TTexchanged.tex", format="pgf")
        plt.show()

                # Create a figure with a specific size and resolution

        keys = list(HybridSwitching.keys())
        values1 = list(HybridSwitching.values())
        values2 = [i/1e-9 for i in list(HybridEnergy.values())]
        fig, ax1 = plt.subplots(figsize=(10, 8))

        # Bar plot for the first dictionary
        bar_width = 0.4
        x_positions = range(len(keys))  # X positions for bars
        bars = ax1.bar(x_positions, values1, color='green', width=bar_width, label='Tech. Switching')
        ax1.set_ylabel('Avg. Technology Switching/Node')
        ax1.tick_params(axis='y')
        ax1.set_xlabel('Avg. Number of Packets/Node')
        step = 5  # Every 10th position
        ticks = x_positions[::step]  # Every 10th position
        tick_labels = [f'{keys[i]-1:.0f}' for i in range(0, len(keys), step)]  # Label for every 10th position
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(tick_labels)  # Set labels at the correct positions
        # Add a grid to the bar plot
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Create a second y-axis for the line plot
        ax2 = ax1.twinx()
        ax2.plot(x_positions, values2, color='orange', marker='o', linestyle='-', label='Energy Consumption', linewidth=2)
        ax2.set_ylabel('Avg. Energy Consumption/Com. Node (nJ)')
        ax2.tick_params(axis='y')

        # Add a grid to the line plot
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Title and legends
        ax1.legend(loc='lower right', fontsize=15)
        ax2.legend(loc='upper right', fontsize=15)

        # Tidy layout for better visualization
        fig.tight_layout()
        #plt.savefig("Switching_TTexchanged.tex", format="pgf")
        #tikzplotlib.save("Switching_TTexchanged.tex")

        # Show the plot
        plt.show()
 
   
            
    # 2. Mean AoI plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate mean values over Monte Carlo runs
    MAOI_RF = [MAoIRF[key]/N_montecarlo for key in sorted_keys]
    MAOI_Hybrid = [MAoIHybrid[key]/N_montecarlo for key in sorted_keys]
    MAOI_WOC = [MAoIWOC[key]/N_montecarlo for key in sorted_keys]
    MAOI_RF_C = [MAoIRF_C[key]/N_montecarlo for key in sorted_keys]
    MAOI_Hybrid_C = [MAoIHybrid_C[key]/N_montecarlo for key in sorted_keys]
    MAOI_WOC_C = [MAoIWOC_C[key]/N_montecarlo for key in sorted_keys]
    
    # Plot mean AoI with explicit line objects
    lines = []
    lines.append(ax.plot(sorted_keys, MAOI_RF, color='red', linestyle='--', label='M-AoI RF')[0])
    lines.append(ax.plot(sorted_keys, MAOI_Hybrid, color='orange', linestyle='--', label='M-AoI RF-WOC')[0])
    lines.append(ax.plot(sorted_keys, MAOI_WOC, color='grey', linestyle='--', label='M-AoI WOC')[0])
    lines.append(ax.plot(sorted_keys, MAOI_RF_C, color='green', linestyle='--', label='M-AoI RF (Communicating)')[0])
    lines.append(ax.plot(sorted_keys, MAOI_Hybrid_C, color='black', linestyle='--', label='M-AoI RF-WOC (Communicating)')[0])
    lines.append(ax.plot(sorted_keys, MAOI_WOC_C, color='orange', linestyle='--', label='M-AoI WOC (Communicating)')[0])
    
    if Type == 'Energy/Throughput':
        ax.set_xlabel('Avg. Number Of Packets Per Communication')
    elif Type == 'N_APs':
        ax.set_xlabel('Number Of APs')
    elif Type == 'N_d':
        ax.set_xlabel('Number Of Devices')
    ax.set_ylabel('Age Of Information')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    
    ax.grid(which='major', axis='both', linestyle='--', alpha=0.7, color='gray', linewidth=0.8)
    ax.grid(which='minor', axis='both', linestyle=':', alpha=0.5, color='gray', linewidth=0.5)
    
    # Create legend with explicit lines
    ax.legend(lines, [line.get_label() for line in lines], loc='center right', fontsize=15)
    plt.show()
    
    # 3. Peak AoI plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate peak values over Monte Carlo runs
    PAOI_RF = [PAoIRF[key]/N_montecarlo for key in sorted_keys]
    PAOI_Hybrid = [PAoIHybrid[key]/N_montecarlo for key in sorted_keys]
    PAOI_WOC = [PAoIWOC[key]/N_montecarlo for key in sorted_keys]
    PAOIRF_C = [PAoIRF_C[key]/N_montecarlo for key in sorted_keys]
    PAOIHybrid_C = [PAoIHybrid_C[key]/N_montecarlo for key in sorted_keys]
    PAOIWOC_C = [PAoIWOC_C[key]/N_montecarlo for key in sorted_keys]
    
    # Plot peak AoI with explicit line objects
    lines = []
    lines.append(ax.plot(sorted_keys, PAOIRF_C, color='blue', linestyle='--', label='P-AoI RF (Communicating)')[0])
    lines.append(ax.plot(sorted_keys, PAOIHybrid_C, color='cyan', linestyle='--', label='P-AoI RF-WOC (Communicating)')[0])
    lines.append(ax.plot(sorted_keys, PAOIWOC_C, color='green', linestyle='--', label='P-AoI WOC (Communicating)')[0])
    lines.append(ax.plot(sorted_keys, PAOI_RF, color='grey', linestyle='--', label='P-AoI RF')[0])
    lines.append(ax.plot(sorted_keys, PAOI_Hybrid, color='purple', linestyle='--', label='P-AoI RF-WOC')[0])
    lines.append(ax.plot(sorted_keys, PAOI_WOC, color='yellow', linestyle='--', label='P-AoI WOC')[0])
    
    if Type == 'Energy/Throughput':
        ax.set_xlabel('Avg. Number Of Packets Per Communication')
    elif Type == 'N_APs':
        ax.set_xlabel('Number Of APs')
    elif Type == 'N_d':
        ax.set_xlabel('Number Of Devices')
    ax.set_ylabel('Age Of Information')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.grid(which='major', axis='both', linestyle='--', alpha=0.7, color='gray', linewidth=0.8)
    ax.grid(which='minor', axis='both', linestyle=':', alpha=0.5, color='gray', linewidth=0.5)
    
    # Create legend with explicit lines
    ax.legend(lines, [line.get_label() for line in lines], loc='center right', fontsize=15)
    plt.show()

   

def plot_scheduling_matrix(scheduling_matrix, N, N_APs, Time):
    """Plot scheduling matrix visualization."""
    # Set more appropriate font sizes
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 8
    })

    plotSize = N
    fig, axes = plt.subplots(plotSize, plotSize, figsize=(15, 15))
    
    # Create custom colormap
    cmap = ListedColormap(["white", "blue", "orange"])  # White for no usage, Blue for RF, Orange for WOC
    
    for i in range(plotSize):
        Text1 = 'AP' if i < N_APs else 'Node'
        for j in range(plotSize):
            # Extract RF and WOC usage
            rf_usage = scheduling_matrix[i, j, 0, :]  # RF starts at index 0
            woc_usage = scheduling_matrix[i, j, 1, :].reshape(1, -1)  # WOC is at index 1
    
            # Create combined usage matrix
            combined_usage = np.vstack([woc_usage * 2, rf_usage])  # Scale WOC values to distinguish it
    
            # Plot combined heatmap
            ax = axes[i, j]
            sns.heatmap(
                combined_usage,
                cmap=cmap,
                cbar=False,
                ax=ax,
                xticklabels=range(Time),
                yticklabels=['WOC', "RF"],
                vmin=0, vmax=2,  # Ensure proper mapping to colormap
                square=True  # Make the heatmap square
            )
    
            Text2 = 'AP' if j < N_APs else 'Node'
            
            # Set title and labels with improved formatting
            ax.set_title(f"{Text1} {i} → {Text2} {j}", pad=5)
            ax.set_xlabel('Time step', labelpad=2)
            ax.set_ylabel('Technology', labelpad=2)
            
            # Adjust tick parameters
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xticks(np.arange(Time))
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(['WOC', 'RF'])
    
    # Adjust layout to prevent label overlap
    plt.tight_layout(pad=2.0)
    
    # Ensure the figure is displayed
    plt.show(block=True)
        


def AoI(t,selection,intervals,Linkratiosent,typ='exp',VIZ=False,Verbose=False):
    # Parameters
    # Determine the simulation parameters
    input_data=[]
    for i in range(0,len(selection)):
        input_data.append([selection[i][0],selection[i][1],selection[i][2]-intervals[i]+1,selection[i][2]+transferTime,selection[i][4],selection[i][5]])
        
      
    # Identify unique sender-receiver pairs
    pairs = set((sender, receiver) for sender, receiver, _, _ ,_,_ in input_data)


    # Initialize simulation parameters
    total_time = t+1
    time_step = 0.01
    time_vector = np.arange(0, total_time, time_step)
    mapping_m=defaultdict(lambda: 0)
    mapping_p=defaultdict(lambda: 0)


    # Initialize AoI matrix for each pair
    aoi_matrices = {pair: np.full(len(time_vector), np.inf) for pair in pairs}
    Technology = {pair: -1 for pair in pairs}

    # Process packet receptions to update AoI for each pair
    for sender, receiver, generated_time, received_time, technology, whichmsg in sorted(input_data, key=lambda x: x[3]):
        pair = (sender, receiver)
        reception_index = np.searchsorted(time_vector, received_time)
        Technology[pair]= 2 if Technology[pair]!=-1 and Technology[pair]!=technology else technology
        if reception_index < len(time_vector):
            # Calculate AoI based on the current packet
            current_aoi = received_time - generated_time
            aoi_matrices[pair][reception_index] = min(aoi_matrices[pair][reception_index], current_aoi)
            # Increment AoI linearly from this point
            if typ=='exp':
                for idx in range(0,reception_index):
                    if aoi_matrices[pair][idx]==np.inf:
                        aoi_matrices[pair][idx] = np.exp(idx* time_step)
                        
                        
                for idx in range(reception_index + 1, len(time_vector)):
                    aoi_matrices[pair][idx] = np.exp(aoi_matrices[pair][idx - 1]+ time_step)
            else:
                for idx in range(0,reception_index):
                    if aoi_matrices[pair][idx]==np.inf:
                        aoi_matrices[pair][idx] = idx* time_step
                
                for idx in range(reception_index + 1, len(time_vector)):
                    aoi_matrices[pair][idx] = aoi_matrices[pair][idx - 1] + time_step
            mapping_m[sender,receiver,whichmsg]=np.mean(aoi_matrices[pair][:])
            mapping_p[sender,receiver,whichmsg]=np.max(aoi_matrices[pair][:])
    if VIZ:

        plt.figure(figsize=(14, 8))
        for pair, aoi_matrix in aoi_matrices.items():
            sender, receiver = pair
            if Technology[pair]==0:
                plt.plot(time_vector, aoi_matrix, marker = next(marker),markersize=15, markevery=100,     linewidth=2,   label=f'From {sender} to {receiver} - RF')
            elif Technology[pair]==1:
                plt.plot(time_vector, aoi_matrix, marker = next(marker),markersize=15, markevery=100,     linewidth=3,   label=f'From {sender} to {receiver} - WOC')
            elif Technology[pair]==2:
                plt.plot(time_vector, aoi_matrix, marker = next(marker),markersize=15, markevery=100,     linewidth=4,   label=f'From {sender} to {receiver} - RF/WOC')
            else:
                print('ERROR')
                return

        plt.xlabel('Time (unit)')
        plt.ylabel('AoI Evolution')
        plt.hlines(y =1, xmin = 0, xmax = t+1,color = 'black', linewidth=5, linestyle = '--', label='Minimum AOI') 
        plt.legend()
        plt.grid(True)
        plt.show()

         # Add bar plot for average AoI per pair
    
    pairs_ = []
    avg_aoi = []
    colors = []
    
    for pair, aoi_matrix in aoi_matrices.items():
        sender, receiver = pair
        # Calculate average AoI excluding inf values
        valid_aoi = aoi_matrix[np.isfinite(aoi_matrix)]
        avg = np.mean(valid_aoi)
        pairs_.append(f'{sender}→{receiver}')
        avg_aoi.append(avg)
        
        # Set color based on technology
        if Technology[pair] == 0:
            colors.append('red')  # RF
        elif Technology[pair] == 1:
            colors.append('orange')  # WOC
        else:
            colors.append('green')  # RF/WOC
    
    if VIZ:# Create bar plot
        plt.figure(figsize=(14, 8))
        bars = plt.bar(pairs_, avg_aoi, color=colors)
        plt.xlabel('Sender-Receiver Pairs')
        plt.ylabel('Average AoI')
        plt.title('Average Age of Information per Pair')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        
        # Add legend for technology types
        legend_elements = [
            Patch(facecolor='red', label='RF'),
            Patch(facecolor='orange', label='WOC'),
            Patch(facecolor='green', label='RF/WOC')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.show()
        
    peak_aoi_per_pair = {}
    mean_aoi_per_pair = {}
    # Calculate peak and mean AoI for each sender-receiver pair
    for pair, aoi_matrix in aoi_matrices.items():
        # Assuming AoI starts counting after the first packet is received,
        # we exclude the initial np.inf values for accurate calculations
        valid_aoi_values = aoi_matrix[np.isfinite(aoi_matrix)]

        # Calculate and store peak AoI
        peak_aoi_per_pair[pair] = np.max(valid_aoi_values)
        # Calculate and store mean AoI
        mean_aoi_per_pair[pair] = np.mean(valid_aoi_values)
  
        
    # Print the calculated metrics for each sender-receiver pair
    if Verbose:
        for pair in pairs:
            print(f"Sender {pair[0]} to Receiver {pair[1]} - Peak AoI: {peak_aoi_per_pair[pair]}, Mean AoI: {mean_aoi_per_pair[pair]}")
    meanSendingPairs=np.array(list(mean_aoi_per_pair.values())).mean()  
    peakSendingPairs=np.array(list(peak_aoi_per_pair.values())).mean()

    allAoI_m= {}
    allAoI_p= {}

    for (i, j, nmessage, k), value in Linkratiosent.items():  
        allAoI_m[i,j,nmessage]=(1 - value) *t + value*mapping_m[i,j,nmessage]
        allAoI_p[i,j,nmessage]=(1 - value) *t + value*mapping_p[i,j,nmessage]

    meanALLPairs=np.array(list(allAoI_m.values())).mean()
    peakallPairs=np.array(list(allAoI_p.values())).mean()
    
    return peakSendingPairs,meanSendingPairs,peakallPairs,meanALLPairs
    
def AoIperApp(N,t,selection,intervals,typ='exp',VIZ=False,Verbose=False):
    

   howManyapp=set([selection[i][3] for i in range(0,len(selection))])
    # Parameters
    # Determine the simulation parameters
   for app in list(howManyapp):
        input_data=[]
        for i in range(0,len(selection)):
            input_data.append([selection[i][0],selection[i][1],selection[i][2]-intervals[i]+1,selection[i][2]+transferTime,selection[i][4]])
            
        
        
        # Identify unique sender-receiver pairs
        pairs = set((sender, receiver) for sender, receiver, _, _ ,_ in input_data)
        all_pairs = {(i, j) for i in range(N + 1) for j in range(N + 1) if i != j}
        # Initialize simulation parameters
        total_time = t+1        
        time_step = 0.01
        time_vector = np.arange(0, total_time, time_step)
        
        # Initialize AoI matrix for each pair
        aoi_matrices = {pair: np.full(len(time_vector), np.inf) for pair in pairs}
        Technology = {pair: -1 for pair in pairs}
        
        # Process packet receptions to update AoI for each pair
        for sender, receiver, generated_time, received_time, technology in sorted(input_data, key=lambda x: x[3]):
            pair = (sender, receiver)
            reception_index = np.searchsorted(time_vector, received_time)
            Technology[pair]= 2 if Technology[pair]!=-1 and Technology[pair]!=technology else technology
            if reception_index < len(time_vector):
                # Calculate AoI based on the current packet
                current_aoi = received_time - generated_time
                aoi_matrices[pair][reception_index] = min(aoi_matrices[pair][reception_index], current_aoi)
                # Increment AoI linearly from this point
                if typ=='exp':
                    for idx in range(0,reception_index):
                        if aoi_matrices[pair][idx]==np.inf:
                            aoi_matrices[pair][idx] = np.exp(idx* time_step)
                            
                            
                    for idx in range(reception_index + 1, len(time_vector)):
                        aoi_matrices[pair][idx] = np.exp(aoi_matrices[pair][idx - 1]+ time_step)
                else:
                    for idx in range(0,reception_index):
                        if aoi_matrices[pair][idx]==np.inf:
                            aoi_matrices[pair][idx] = idx* time_step
                    
 
        if VIZ:
        # Plot AoI for each sender-receiver pair
            plt.figure(figsize=(14, 8))
            for pair, aoi_matrix in aoi_matrices.items():
                sender, receiver = pair
                plt.plot(time_vector, aoi_matrix, marker = next(marker), markersize=15, markevery=100,     linewidth=2, label=f'From {sender} to {receiver} - App {app}')
            
            plt.xlabel('Time (unit)')
            plt.ylabel('AoI Evolution (Per App)')
            plt.hlines(y =1, xmin = 0, xmax = t+1,color = 'black', linewidth=5, linestyle = '--', label='Minimum AOI') 
            plt.legend()
            plt.grid(True)
            plt.show()
        
        peak_aoi_per_pair = {}
        mean_aoi_per_pair = {}
        
        # Calculate peak and mean AoI for each sender-receiver pair
        for pair, aoi_matrix in aoi_matrices.items():
            # Assuming AoI starts counting after the first packet is received,
            # we exclude the initial np.inf values for accurate calculations
            valid_aoi_values = aoi_matrix[np.isfinite(aoi_matrix)]
            
            # Calculate and store peak AoI
            peak_aoi_per_pair[pair] = np.max(valid_aoi_values)
            
            # Calculate and store mean AoI
            mean_aoi_per_pair[pair] = np.mean(valid_aoi_values)
        if Verbose:
            for pair in pairs:
                print(f"Sender {pair[0]} to Receiver {pair[1]} for Application {app} - Peak AoI: {peak_aoi_per_pair[pair]}, Mean AoI: {mean_aoi_per_pair[pair]}")
                    
    


# Rician fading function
def rician_fading(k_factor, size=1):
    """Simulate Rician fading for LoS conditions."""
    K_linear = 10 ** (k_factor / 10)  # Convert K-factor to linear
    s = np.sqrt(K_linear / (K_linear + 1))  # LoS component
    sigma = np.sqrt(1 / (2 * (K_linear + 1)))  # NLoS component
    fading = np.sqrt((s + sigma * np.random.randn(size)) ** 2 + (sigma * np.random.randn(size)) ** 2)
    return fading
# Extract soltuion from Optimization
def getSolution(x):
    solution = pd.DataFrame.from_dict(x, orient="index", columns = ["variable_object"])
    solution.reset_index(inplace=True)
    solution["solution_value"] = solution["variable_object"].apply(lambda item: item.solution_value)
    return solution["solution_value"]

def process_solution(MINLP_solution, opt_model, Verbose):
    """
    Process and print solver details if solution exists.
    
    Args:
        MINLP_solution: Solution object from the solver
        opt_model: The optimization model
        Verbose: Boolean flag to control output verbosity
        
    Returns:
        bool: True if solution exists, False otherwise
    """
    if MINLP_solution:
        if Verbose:
            print("Solver Details:")
            print(f"Status: {opt_model.solve_details.status}")
            print(f"Objective Value: {MINLP_solution.objective_value}")
            print(f"Best Bound: {opt_model.solve_details.best_bound}")
            print(f"Gap: {opt_model.solve_details.mip_relative_gap * 100:.2f}%")
            print(f"Nodes Explored: {opt_model.solve_details.nb_nodes_processed}")
            print(f"Solve Time: {opt_model.solve_details.time:.2f} seconds")
        return True
    else:
        print("No solution found.")
        return False

def get_simulation_type(M_WOC, M):
    """
    Determine the simulation type based on M_WOC and M values.
    
    Args:
        M_WOC (int): Number of WOC interfaces
        M (int): Total number of interfaces
        
    Returns:
        str: Simulation type ('RF_ONLY', 'WOC_ONLY', or 'HYBRID')
    """
    if M_WOC == 0 and M == 2:
        return 'HYBRID'
    elif M_WOC == 0 and M == 1:
        return 'RF_ONLY'
    elif M_WOC == 1 and M == 2:
        return 'WOC_ONLY'
    else:
        raise ValueError(f"Invalid combination of M_WOC={M_WOC} and M={M}")
def create_scheduling_matrix(N, M_WOC, M, Time, Binaryx, T):
    """Create scheduling matrix from solution."""
    Nsatisfied = 0
    indices = []
    Scheduling = []
    scheduling_matrix = np.zeros((N, N, 2, Time))
    
    for i in range(N):
        for j in range(N):
            for m in range(M_WOC, M):
                    for k in range(Time):
                        indices.append([i, j, m, k])
    
    for index, value in enumerate(indices):
        if Binaryx[index] > 0.5:
            Nsatisfied = Nsatisfied + T[value[3]][value[0]][value[1]]
            Scheduling.append(value)
            scheduling_matrix[value[0]][value[1]][value[2]][value[3]] = 1
            
    return Nsatisfied, Scheduling, scheduling_matrix

def process_messages(N, msgQueues, Binaryx_m, Unsent, MsgNumber, S_p):
    """Process message choices and calculate total bits exchanged."""
    indices = []
    choices = []
    ratiosent = defaultdict(lambda: 0)
    Linkratiosent = defaultdict(lambda: 0)
    total_bits_exchanged = 0

    for i in range(N):
        for j in range(N):
            for nmessage in range(0, len(msgQueues[i][j])):
                indices.append([i, j, nmessage])
                for k in set(msgQueues[i][j][nmessage]):
                    Linkratiosent[i,j,nmessage,k] = 0
        
    for index, value in enumerate(indices):
        if Binaryx_m[index] > 0.5:
            choices.append(value)
            total_bits_exchanged += (MsgNumber[value[0]][value[1]][value[2]] - Unsent[index]) * S_p
            ratiosent[value[0],value[1],value[2]] = (MsgNumber[value[0]][value[1]][value[2]] - Unsent[index])/(MsgNumber[value[0]][value[1]][value[2]])
    
    return choices, ratiosent, Linkratiosent, total_bits_exchanged

def calculate_energy_consumption(N, M_WOC, M, Time, Binaryx, SendEnergy, RecieveEnergy):
    """Calculate total energy consumption."""
    consumed_energy = 0
    indices = []

    for i in range(N):
        for j in range(N):
            for m in range(M_WOC, M):
                    for k in range(Time):
                        indices.append([i, j, m, k])
    
    for index, value in enumerate(indices):
        if Binaryx[index] > 0.5:
            if value[2] > 0.5:  # WOC
                consumed_energy += SendEnergy[1][value[0]][value[1]][value[3]] + RecieveEnergy[1][value[0]][value[1]][value[3]]
            else:  # RF
                consumed_energy += SendEnergy[0][value[0]][value[1]][value[3]] + RecieveEnergy[0][value[0]][value[1]][value[3]]

    return consumed_energy
def run_simulation(M_WOC, M, maxN, N, Time, msgQueues, BiggestMsg, MsgNumber, T,
                  snr_db_matrix_RF, snr_db_matrix_WOC, snr_min_rf, snr_min_woc,
                  capacity_matrix, S_p, Time_unit, SendEnergy, RecieveEnergy, EnergyTotal,
                  Verbose, VIZ, objective_type='max_messages', Timeout=60, msgApp=None, N_APs=None):
    """
    Run a single simulation with specified parameters.
    
    Args:
        M_WOC (int): Number of WOC interfaces
        M (int): Total number of interfaces
        maxN (int): Maximum number of messages
        N (int): Total number of nodes
        Time (int): Number of time slots
        msgQueues (list): Message queues for each node pair
        BiggestMsg (int): Size of largest message
        MsgNumber (list): Number of messages for each node pair
        T (list): Time slots for each message
        snr_db_matrix_RF (numpy.ndarray): SNR matrix for RF
        snr_db_matrix_WOC (numpy.ndarray): SNR matrix for WOC
        snr_min_rf (float): Minimum SNR for RF
        snr_min_woc (float): Minimum SNR for WOC
        capacity_matrix (numpy.ndarray): Capacity matrix
        S_p (int): Packet size
        Time_unit (float): Time unit duration
        SendEnergy (list): Energy required for sending
        RecieveEnergy (list): Energy required for receiving
        EnergyTotal (float): Total available energy
        Verbose (bool): Whether to print detailed information
        VIZ (bool): Whether to plot results
        N_APs (int): Number of access points
        objective_type (str): Type of objective to optimize (default: 'max_messages')
        Timeout (int): Time limit for solving the model (default: 60 seconds)
        
    Returns:
        dict: Dictionary containing simulation results
    """
    sim_type = get_simulation_type(M_WOC, M)
    if Verbose:
        print(f"\nRunning {sim_type} simulation with objective: {objective_type}...")
    allowedTechnology=np.zeros((Time,N,N))
    ##Create labeled dataset for Seq-GTN
    if sim_type == 'HYBRID':
        NodesF=[]
        for k in range(0,Time):
            for i in range(N):
                NodesF.append([k,i,float(EnergyTotal[i])])
        c = open('./dataset/inputNodesFeatures'+str(N)+'.csv', 'a+')
        cw = csv.writer(c, lineterminator='')
        c.seek(0, 2)  # Move to end of file
        if c.tell() == 0:  # Check if file is empty
            cw.writerow(['TimeStep', 'NodeID', 'EnergyLevel'])
            cw.writerow('\n')
        for nodeF in NodesF:
            cw.writerow(nodeF)
            cw.writerow('\n')
        c.close()
                

        for k in range(0,Time):
                for i in range(0,N):
                    for j in range(0,N):
                        if i!=j:
                            if snr_db_matrix_RF[i][j][k]>=snr_min_rf and snr_db_matrix_WOC[i][j][k]>=snr_min_woc:
                                allowedTechnology[k][i][j]=3
                            elif snr_db_matrix_RF[i][j][k]>=snr_min_rf:
                                allowedTechnology[k][i][j]=1
                            elif snr_db_matrix_WOC[i][j][k]>=snr_min_woc:
                                allowedTechnology[k][i][j]=2
                            else:
                                allowedTechnology[k][i][j]=0

        c = open('./dataset/inputEdgesFeatures'+str(N)+'.csv', 'a+')
        cw = csv.writer(c, lineterminator='')
        c.seek(0, 2)  # Move to end of file
        if c.tell() == 0:  # Check if file is empty
            cw.writerow(['TimeStep', 'SourceNode', 'TargetNode', 'SendingStatus', 
                        'RF_SendEnergy', 'RF_ReceiveEnergy', 'WOC_SendEnergy', 'WOC_ReceiveEnergy',
                        'MessageQueueLength', 'AllowedTechnology'])
            cw.writerow('\n')
        for k in range(0,Time):
            for i in range(0,N):
                for j in range(0,N):
                    if i!=j:
                        cw.writerow([k,i,j,T[k][i][j],SendEnergy[0][i][j][k],RecieveEnergy[0][i][j][k],SendEnergy[1][i][j][k],RecieveEnergy[1][i][j][k],len(msgQueues[i][j]),allowedTechnology[k][i][j]])
                        cw.writerow('\n')
        c.close()

    # Run MINLP model
    MINLP_solution, opt_model, x, x_m, status, delta, Unsent = MINLP_model(
        M_WOC, M, maxN, N, Time, msgQueues, BiggestMsg, MsgNumber, T,
        snr_db_matrix_RF, snr_db_matrix_WOC, snr_min_rf, snr_min_woc,
        capacity_matrix, S_p, Time_unit, SendEnergy, RecieveEnergy, EnergyTotal,
        objective_type=objective_type, Timeout=Timeout
    )
    
    
    if not process_solution(MINLP_solution, opt_model, Verbose):
        return None
    
    # Process solution
    Unsent = getSolution(Unsent)
    Binaryx = getSolution(x)
    Binaryx_m = getSolution(x_m)
    
    # Create scheduling matrix
    Nsatisfied, Scheduling, scheduling_matrix = create_scheduling_matrix(N, M_WOC, M, Time, Binaryx, T)
    
    if VIZ:
        plot_scheduling_matrix(scheduling_matrix, N, N_APs, Time)
    
    # Process messages
    choices, ratiosent, Linkratiosent, total_bits_exchanged = process_messages(
        N, msgQueues, Binaryx_m, Unsent, MsgNumber, S_p
    )

    Scheduling.sort(key=lambda x: (x[0], x[1], x[3], x[2]))
    if len(Scheduling) == 0:
        return None
    if len(Scheduling) != len(choices):
        raise Exception("Dimension not matching! Debug this.")

    # Calculate energy consumption
    consumed_energy = calculate_energy_consumption(N, M_WOC, M, Time, Binaryx, SendEnergy, RecieveEnergy)
    if Verbose:
        print(f'Consumed {sim_type} Energy in Joule {consumed_energy}')
        print(f'Avg. Consumed {sim_type} Energy in Joule {consumed_energy/len(Scheduling)}')
    
    # Calculate delay
    MaxDelay = 0
    Delay = 0
    CommunicatingDevices = defaultdict(lambda: 0)
    
    for st in range(0, len(Scheduling)):
        i, j, m, k = Scheduling[st]
        i, j, f = choices[st]
        CommunicatingDevices[i,j,f] = 1
        Linkratiosent[i,j,f,k] = ratiosent[i,j,f]
        Delay = Delay + np.argwhere(np.array(msgQueues[i][j][f]) == k)[0][0]+1
        MaxDelay = MaxDelay + len(np.array(msgQueues[i][j][f]))
    
    if MaxDelay == 0:
        Delay = 0
        print('Messages sent with 0 delay!')
    else:
        print(f'Messages sent with {Delay/MaxDelay} delay!')
    
    # Calculate total delay
    MaxTotalDelay = 0
    TotalDelay = 0
            
    for i in range(N):
        for j in range(N):
            for f in range(0, len(msgQueues[i][j])):
                if CommunicatingDevices[i,j,f] != 1:
                    MaxTotalDelay = MaxTotalDelay + BiggestMsg + 1
                    TotalDelay = TotalDelay + BiggestMsg + 1

    MaxTotalDelay += MaxDelay
    TotalDelay += Delay
    
    if Verbose:
        print(f'Overall Platform delay is {TotalDelay/MaxTotalDelay}')
        
    # Process status and switching
    statusSol = getSolution(status)
    indices = []
    switching = defaultdict(list)
    nodeswitch = defaultdict(lambda: 0)
    
    for i in range(N):
        for k in range(Time):
            indices.append([i, k])
    
    for index, value in enumerate(indices):
        if statusSol[index] > 0.9:
                switching[value[0], value[1]] = 1
        elif statusSol[index] < 0.1:
                switching[value[0], value[1]] = 0
                
    Nswitch = 0
    for i in range(0, N):
        nodeswitch[i] = 0
        if switching[i, 0] == 1:
            nodeswitch[i] += 1
            Nswitch = Nswitch + 1
        for k in range(1, Time):
            if switching[i, k] != switching[i, k-1]:
                Nswitch = Nswitch + 1
                if i in nodeswitch:
                    nodeswitch[i] += 1
                else:
                    nodeswitch[i] = 1
    
    Nswitch = Nswitch/(N)

    if Verbose:
        print('Average Switching between RF and WOC Per node, ', Nswitch)
        
    if VIZ:
            plt.figure(figsize=(6, 6))
            node_ids, switches = zip(*sorted(nodeswitch.items()))
            node_ids = tuple(x + 1 for x in node_ids)
            plt.bar(node_ids, switches)
            plt.xlabel("Node ID")
            plt.ylabel("Number of Switches Per Node")
            plt.xticks(range(int(min(node_ids)), int(max(node_ids)) + 1))
            plt.yticks(range(0, int(max(switches)) + 1))
            plt.show()
        
    # Process delta and AoI
    Delta_resultat = getSolution(delta)
    indices = []
    update_intervals = []
    selection = []
    
    for i in range(N):
        for j in range(N):
            for f in range(0, len(msgQueues[i][j])):
                indices.append([i, j, f])
    
    for index, value in enumerate(indices):
        if Delta_resultat[index] <= 0.5:
            raise Exception('ERROR, Debug delay, this shouldnt happen')
        if Delta_resultat[index] > 0.5 and Delta_resultat[index] < (BiggestMsg+1-0.2):
            selection.append(value)
            update_intervals.append(Delta_resultat[index])
            
    appselection = []
    for i in range(0, len(choices)):
        appselection.append([
            choices[i][0], choices[i][1], Scheduling[i][3],
            msgApp[choices[i][0]][choices[i][1]][choices[i][2]][0],
            Scheduling[i][2], choices[i][2]
        ])
    
    if len(selection) != 0:
        peak_aoi_communicating, mean_aoi_communicating, peak_aoi_all, mean_aoi_all = AoI(
            Time, appselection, update_intervals, Linkratiosent, "Lin", VIZ, Verbose
        )
        AoIperApp(N, Time, appselection, update_intervals, "Lin", VIZ, Verbose)
        
    if Verbose:
        print(f'The Peak AoI for communicating pairs using {sim_type} is {peak_aoi_communicating}')
        print(f'The Mean AoI for communicating pairs using {sim_type} is {mean_aoi_communicating}')
        print(f'The Peak AoI for all pairs using {sim_type} is {peak_aoi_all}')
        print(f'The Mean AoI for all pairs using {sim_type} is {mean_aoi_all}')
    if sim_type == 'HYBRID':
        nodes = {i for i in range(0,N)}  # Replace with all unique nodes in the network
        full_time_steps = range(0, Time)


        # Initialize DataFrames for RF and Optical
        df_energy = pd.DataFrame(index=full_time_steps, columns=list(nodes))
        
        # Initialize with initial energy levels for all nodes
        for node in nodes:
            df_energy.loc[0, node] = EnergyTotal[node]

        # Process each time step
        for time_step in full_time_steps:
            # Carry over energy levels from the previous time step
            if time_step > 0:
                df_energy.loc[time_step] = df_energy.loc[time_step - 1]

            # Filter communication logs for the current time step
            current_logs = [log for log in Scheduling if log[3] == time_step]

            # Update energy levels based on communication
            for log in current_logs:
                sender, receiver, technology_id, _ = log

                if technology_id == 0:  # RF
                        df_energy.loc[time_step, sender] -= SendEnergy[0][sender][receiver][time_step]
                        df_energy.loc[time_step, receiver] -= RecieveEnergy[0][sender][receiver][time_step]
                elif technology_id == 1:  # Optical
                        df_energy.loc[time_step, sender] -= SendEnergy[1][sender][receiver][time_step]
                        df_energy.loc[time_step, receiver] -= RecieveEnergy[1][sender][receiver][time_step]

        # Save energy data in three-column format
        energy_data = []
        for time_step in full_time_steps:
            for node in nodes:
                energy_data.append([time_step, node, df_energy.loc[time_step, node]])
        
        # Write to CSV with headers (only if file is empty)
        with open('./dataset/recordedNodesFeatures'+str(N)+'.csv', 'a+') as f:
            # Check if file is empty to determine if we need headers
            f.seek(0, 2)  # Move to end of file
            write_header = f.tell() == 0  # Check if file is empty
            
            writer = csv.writer(f, lineterminator='\n')
            if write_header:
                writer.writerow(['TimeStep', 'NodeID', 'EnergyLevel'])
            writer.writerows(energy_data)

        thePossibleCOm=0
        newLABEL=0
        c = open('./dataset/recordedEdgesFeatures'+str(N)+'.csv', 'a+')
        cw = csv.writer(c, lineterminator='')
        c.seek(0, 2)  # Move to end of file
        if c.tell() == 0:  # Check if file is empty
            cw.writerow(['TimeStep', 'SourceNode', 'TargetNode', 'SendingStatus', 
                        'QueueLength', 'AllowedTechnology', 'SelectedTechnology', 'Label'])
            cw.writerow('\n')
        for k in range(0,Time):
            for i in range(0,N):
                for j in range(0,N):
                        if i!=j:
                            selected=0
                            for selection in Scheduling:
                                if selection[3]==k and selection[0]==i and selection[1]==j:
                                    selected=selection[2]+1
                            thePossibleCOm=allowedTechnology[k][i][j]
                            if thePossibleCOm==0 and selected ==0:
                                newLABEL=0
                            elif thePossibleCOm==1 and selected ==0:
                                newLABEL=1
                            elif thePossibleCOm==1 and selected ==1:
                                newLABEL=2
                            elif thePossibleCOm==2 and selected ==0:
                                newLABEL=3
                            elif thePossibleCOm==2 and selected ==2:
                                newLABEL=4
                            elif thePossibleCOm==3 and selected ==0:
                                newLABEL=5
                            elif thePossibleCOm==3 and selected ==1:
                                newLABEL=6
                            elif thePossibleCOm==3 and selected ==2:
                                newLABEL=7
                            cw.writerow([k,i,j,T[k][i][j],len(msgQueues[i][j]),allowedTechnology[k][i][j],selected,newLABEL])
                            cw.writerow('\n')
        
    # Return results
    return {
        'type': sim_type,
        'energy': consumed_energy,
        'total_bits_exchanged': total_bits_exchanged,
        'delay': TotalDelay/MaxTotalDelay,
        'switching': Nswitch,
        'peak_aoi_communicating': peak_aoi_communicating,
        'mean_aoi_communicating': mean_aoi_communicating,
        'peak_aoi_all': peak_aoi_all,
        'mean_aoi_all': mean_aoi_all
    }

def run_APs_simulation(N_APs_range=None, N_d=5, fixed_size=128, 
                              minD=5, maxD=20, minE=0.05, maxE=0.1, Time=10, N_montecarlo=2,
                              Verbose=True, VIZ=True, objective_type='max_messages', Timeout=60,snr_min_rf=30,snr_min_woc=10,S_p=128):
    """
    Run simulations with varying number of APs and fixed number of devices.
    
    Args:
        N_APs_range (list): Range of N_APs values to test
        N_d (int): Fixed number of devices
        fixed_size (int): Fixed packet size in bits
        minD, maxD (float): Min and max distances for network generation
        minE, maxE (float): Min and max energy values for network generation
        Time (int): Number of time slots
        N_montecarlo (int): Number of Monte Carlo simulations
        Verbose (bool): Whether to print detailed information
        VIZ (bool): Whether to plot results
        objective_type (str): Type of objective to optimize
        Timeout (int): Time limit for solving the model
    """
    # Initialize dictionaries to store accumulated results
    EnergyPerBit_Hybrid = defaultdict(lambda: 0)       
    EnergyPerBit_RF = defaultdict(lambda: 0)       
    EnergyPerBit_WOC = defaultdict(lambda: 0)
    TotalExchangedPackagesRF = defaultdict(lambda: 0)
    TotalExchangedPackagesHybrid = defaultdict(lambda: 0)
    TotalExchangedPackagesWOC = defaultdict(lambda: 0)
    MAoIRF = defaultdict(lambda: 0)
    MAoIHybrid = defaultdict(lambda: 0)
    MAoIWOC = defaultdict(lambda: 0)
    PAoIRF = defaultdict(lambda: 0)
    PAoIHybrid = defaultdict(lambda: 0)
    PAoIWOC = defaultdict(lambda: 0)
    MAoIRF_C = defaultdict(lambda: 0)
    MAoIHybrid_C = defaultdict(lambda: 0)
    MAoIWOC_C = defaultdict(lambda: 0)
    PAoIRF_C = defaultdict(lambda: 0)
    PAoIHybrid_C = defaultdict(lambda: 0)
    PAoIWOC_C = defaultdict(lambda: 0)
    HybridSwitching = defaultdict(lambda: 0)
    HybridEnergy = defaultdict(lambda: 0)
    # Run simulations for each network size
    for _NC in range(N_montecarlo):
        if Verbose:
            print(f"\nRunning Monte Carlo simulation {_NC + 1}/{N_montecarlo}")
        
        for N_APs in N_APs_range:
            N = N_APs + N_d
            
            # Create network characteristics
            network = data_generation.NetworkCharacteristics(numAPs=N_APs, numDevices=N_d, num_time_slots=Time)
            
            # Generate network parameters
            (distance_matrix, snr_db_matrix_WOC, capacity_matrix_WOC,
             snr_db_matrix_RF, capacity_matrix_RF, snr_matrix, capacity_matrix,
             SendEnergy, RecieveEnergy, EnergyTotal) = network.generate_network_parameters(
                minD=minD, maxD=maxD, minE=minE, maxE=maxE
            )
            
            if VIZ and _NC == 0:  # Only plot for first Monte Carlo run
                plot_network_parameters(
                    snr_db_matrix_WOC, capacity_matrix_WOC,
                    snr_db_matrix_RF, capacity_matrix_RF,
                    capacity_matrix
                )
            
            # Generate messages
            T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp = network.generate_messages(
                minSize=1, maxSize=3, appSize=2, data_gen_prob=0.7
            )
            
            # Generate message sizes
            MsgSize, MsgNumber, maxN = network.generate_size(
                msgQueues, sizemin=fixed_size, sizemax=fixed_size, S_p=S_p
            )
            
            # Run simulations for each configuration
            results_hybrid = run_simulation(
                M_WOC=0, M=2, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
                BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
                snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC,
                snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
                capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=5e-5,
                SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
                Verbose=Verbose, VIZ=VIZ,
                objective_type=objective_type, Timeout=Timeout, msgApp=msgApp,N_APs=N_APs
            )
            
            results_rf = run_simulation(
                M_WOC=0, M=1, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
                BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
                snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC, 
                snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
                capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=5e-5,
                SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
                Verbose=Verbose, VIZ=VIZ,
                objective_type=objective_type, Timeout=Timeout, msgApp=msgApp, N_APs=N_APs
            )
            
            results_woc = run_simulation(
                M_WOC=1, M=2, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
                BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
                snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC,
                snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
                capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=5e-5,
                SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
                Verbose=Verbose, VIZ=VIZ,
                objective_type=objective_type, Timeout=Timeout, msgApp=msgApp, N_APs=N_APs
            )
            
            # Store results for this Monte Carlo run
            if results_hybrid:
                EnergyPerBit_Hybrid[N_APs]+=results_hybrid['energy']/results_hybrid['total_bits_exchanged']
                TotalExchangedPackagesHybrid[N_APs]+=results_hybrid['total_bits_exchanged']/S_p
                MAoIHybrid[N_APs]+=results_hybrid['mean_aoi_all']
                PAoIHybrid[N_APs]+=results_hybrid['peak_aoi_all']
                MAoIHybrid_C[N_APs]+=results_hybrid['mean_aoi_communicating']
                PAoIHybrid_C[N_APs]+=results_hybrid['peak_aoi_communicating']
                HybridSwitching[N_APs]+=results_hybrid['switching']
                HybridEnergy[N_APs]+=results_hybrid['energy']
            if results_rf:
                EnergyPerBit_RF[N_APs]+=results_rf['energy']/results_rf['total_bits_exchanged']
                TotalExchangedPackagesRF[N_APs]+=results_rf['total_bits_exchanged']/S_p
                MAoIRF[N_APs]+=results_rf['mean_aoi_all']
                PAoIRF[N_APs]+=results_rf['peak_aoi_all']
                MAoIRF_C[N_APs]+=results_rf['mean_aoi_communicating']
                PAoIRF_C[N_APs]+=results_rf['peak_aoi_communicating']
            
            if results_woc:
                EnergyPerBit_WOC[N_APs]+=results_woc['energy']/results_woc['total_bits_exchanged']
                TotalExchangedPackagesWOC[N_APs]+=results_woc['total_bits_exchanged']/S_p
                MAoIWOC[N_APs]+=results_woc['mean_aoi_all']
                PAoIWOC[N_APs]+=results_woc['peak_aoi_all']
                MAoIWOC_C[N_APs]+=results_woc['mean_aoi_communicating']
                PAoIWOC_C[N_APs]+=results_woc['peak_aoi_communicating']
   
    # Plot results using averaged values
    plot_simulation_results(
        EnergyPerBit_RF, EnergyPerBit_Hybrid, EnergyPerBit_WOC,
        TotalExchangedPackagesRF, TotalExchangedPackagesHybrid, TotalExchangedPackagesWOC,
        MAoIRF, MAoIHybrid, MAoIWOC, MAoIRF_C, MAoIHybrid_C, MAoIWOC_C,
        PAoIRF, PAoIHybrid, PAoIWOC, PAoIRF_C, PAoIHybrid_C, PAoIWOC_C, HybridSwitching, HybridEnergy,
        N_montecarlo,Type='N_APs'
    )

def run_Nd_simulation(N_d_range=None, N_APs=5, fixed_size=128, 
                              minD=5, maxD=20, minE=0.05, maxE=0.1, Time=10, N_montecarlo=2,
                              Verbose=True, VIZ=True, objective_type='max_messages', Timeout=60,snr_min_rf=30,snr_min_woc=10,S_p=128):
    """
    Run simulations with varying number of APs and fixed number of devices.
    
    Args:
        N_d_range (list): Range of N_d values to test
        N_APs (int): Fixed number of APs
        fixed_size (int): Fixed packet size in bits
        minD, maxD (float): Min and max distances for network generation
        minE, maxE (float): Min and max energy values for network generation
        Time (int): Number of time slots
        N_montecarlo (int): Number of Monte Carlo simulations
        Verbose (bool): Whether to print detailed information
        VIZ (bool): Whether to plot results
        objective_type (str): Type of objective to optimize
        Timeout (int): Time limit for solving the model
    """
    # Initialize dictionaries to store accumulated results
    EnergyPerBit_Hybrid = defaultdict(lambda: 0)       
    EnergyPerBit_RF = defaultdict(lambda: 0)       
    EnergyPerBit_WOC = defaultdict(lambda: 0)
    TotalExchangedPackagesRF = defaultdict(lambda: 0)
    TotalExchangedPackagesHybrid = defaultdict(lambda: 0)
    TotalExchangedPackagesWOC = defaultdict(lambda: 0)
    MAoIRF = defaultdict(lambda: 0)
    MAoIHybrid = defaultdict(lambda: 0)
    MAoIWOC = defaultdict(lambda: 0)
    PAoIRF = defaultdict(lambda: 0)
    PAoIHybrid = defaultdict(lambda: 0)
    PAoIWOC = defaultdict(lambda: 0)
    MAoIRF_C = defaultdict(lambda: 0)
    MAoIHybrid_C = defaultdict(lambda: 0)
    MAoIWOC_C = defaultdict(lambda: 0)   
    PAoIRF_C = defaultdict(lambda: 0)
    PAoIHybrid_C = defaultdict(lambda: 0)
    PAoIWOC_C = defaultdict(lambda: 0)
    HybridSwitching = defaultdict(lambda: 0)    
    HybridEnergy = defaultdict(lambda: 0)       
    # Run simulations for each network size
    for _NC in range(N_montecarlo):
        if Verbose:
            print(f"\nRunning Monte Carlo simulation {_NC + 1}/{N_montecarlo}")
        
        for N_d in N_d_range:
            N = N_APs + N_d
            
            # Create network characteristics
            network = data_generation.NetworkCharacteristics(numAPs=N_APs, numDevices=N_d, num_time_slots=Time)
            
            # Generate network parameters
            (distance_matrix, snr_db_matrix_WOC, capacity_matrix_WOC,
             snr_db_matrix_RF, capacity_matrix_RF, snr_matrix, capacity_matrix,
             SendEnergy, RecieveEnergy, EnergyTotal) = network.generate_network_parameters(
                minD=minD, maxD=maxD, minE=minE, maxE=maxE
            )
            
            if VIZ and _NC == 0:  # Only plot for first Monte Carlo run
                plot_network_parameters(
                    snr_db_matrix_WOC, capacity_matrix_WOC,
                    snr_db_matrix_RF, capacity_matrix_RF,
                    capacity_matrix
                )
            
            # Generate messages
            T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp = network.generate_messages(
                minSize=1, maxSize=3, appSize=2, data_gen_prob=0.7
            )
            
            # Generate message sizes
            MsgSize, MsgNumber, maxN = network.generate_size(
                msgQueues, sizemin=fixed_size, sizemax=fixed_size, S_p=S_p
            )
            
            # Run simulations for each configuration
            results_hybrid = run_simulation(
                M_WOC=0, M=2, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
                BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
                snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC,
                snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
                capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=5e-5,
                SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
                Verbose=Verbose, VIZ=VIZ,
                objective_type=objective_type, Timeout=Timeout, msgApp=msgApp,N_APs=N_APs
            )
            
            results_rf = run_simulation(
                M_WOC=0, M=1, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
                BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
                snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC, 
                snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
                capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=5e-5,
                SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
                Verbose=Verbose, VIZ=VIZ,
                objective_type=objective_type, Timeout=Timeout, msgApp=msgApp, N_APs=N_APs
            )
            
            results_woc = run_simulation(
                M_WOC=1, M=2, maxN=maxN, N=N, Time=Time, msgQueues=msgQueues,
                BiggestMsg=BiggestMsg, MsgNumber=MsgNumber, T=T,
                snr_db_matrix_RF=snr_db_matrix_RF, snr_db_matrix_WOC=snr_db_matrix_WOC,
                snr_min_rf=snr_min_rf, snr_min_woc=snr_min_woc,
                capacity_matrix=capacity_matrix, S_p=S_p, Time_unit=5e-5,
                SendEnergy=SendEnergy, RecieveEnergy=RecieveEnergy, EnergyTotal=EnergyTotal,
                Verbose=Verbose, VIZ=VIZ,
                objective_type=objective_type, Timeout=Timeout, msgApp=msgApp, N_APs=N_APs
            )
            
            # Store results for this Monte Carlo run
            if results_hybrid:
                EnergyPerBit_Hybrid[N_d]+=results_hybrid['energy']/results_hybrid['total_bits_exchanged']
                TotalExchangedPackagesHybrid[N_d]+=results_hybrid['total_bits_exchanged']/S_p
                MAoIHybrid[N_d]+=results_hybrid['mean_aoi_all']
                PAoIHybrid[N_d]+=results_hybrid['peak_aoi_all']
                MAoIHybrid_C[N_d]+=results_hybrid['mean_aoi_communicating']
                PAoIHybrid_C[N_d]+=results_hybrid['peak_aoi_communicating']
                HybridSwitching[N_d]+=results_hybrid['switching']
                HybridEnergy[N_d]+=results_hybrid['energy']
            
            if results_rf:
                EnergyPerBit_RF[N_d]+=results_rf['energy']/results_rf['total_bits_exchanged']
                TotalExchangedPackagesRF[N_d]+=results_rf['total_bits_exchanged']/S_p
                MAoIRF[N_d]+=results_rf['mean_aoi_all']
                PAoIRF[N_d]+=results_rf['peak_aoi_all']
                MAoIRF_C[N_d]+=results_rf['mean_aoi_communicating']
                PAoIRF_C[N_d]+=results_rf['peak_aoi_communicating']
            
            if results_woc:
                EnergyPerBit_WOC[N_d]+=results_woc['energy']/results_woc['total_bits_exchanged']
                TotalExchangedPackagesWOC[N_d]+=results_woc['total_bits_exchanged']/S_p
                MAoIWOC[N_d]+=results_woc['mean_aoi_all']
                PAoIWOC[N_d]+=results_woc['peak_aoi_all']
                MAoIWOC_C[N_d]+=results_woc['mean_aoi_communicating']
                PAoIWOC_C[N_d]+=results_woc['peak_aoi_communicating']
   
    # Plot results using averaged values
   
    plot_simulation_results(
        EnergyPerBit_RF, EnergyPerBit_Hybrid, EnergyPerBit_WOC,
        TotalExchangedPackagesRF, TotalExchangedPackagesHybrid, TotalExchangedPackagesWOC,
        MAoIRF, MAoIHybrid, MAoIWOC, MAoIRF_C, MAoIHybrid_C, MAoIWOC_C,
        PAoIRF, PAoIHybrid, PAoIWOC, PAoIRF_C, PAoIHybrid_C, PAoIWOC_C, HybridSwitching, HybridEnergy,
        N_montecarlo,Type='N_d'
    )

    
    
    