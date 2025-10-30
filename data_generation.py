# data_generation.py
import numpy as np
import random
from collections import defaultdict
from utils import rician_fading
from constants import (
    q, k_B, Temperature, R_L, IlluminationCoefficient, f_WOC, B_WOC,
    P_tx_WOC, T_optical, R, R_photo, lamda, u, theta, phi, A_rec,
    P_N, V_bias, f_RF, lambda_RF, B_RF, P_tx_RF, attenuation_coefficient_RF,
    d0, PL0, sigma_shadowing, G_tx_RF, G_rx_RF
)
import matplotlib.pyplot as plt

class NetworkCharacteristics:
    """
    A class to generate and manage network characteristics for hybrid RF-WOC networks.
    
    This class handles the generation of:
    - Network topology
    - Channel parameters (SNR, capacity)
    - Energy consumption models
    - Message queues and priorities
    
    Attributes:
        numAPs (int): Number of Access Points
        numDevices (int): Number of IoT devices
        num_time_slots (int): Number of time slots in simulation
    """
    
    def __init__(self, numAPs, numDevices, num_time_slots):
        """
        Initialize network characteristics.
        
        Args:
            numAPs (int): Number of Access Points
            numDevices (int): Number of IoT devices
            num_time_slots (int): Number of time slots
        """
        self.numAPs = numAPs
        self.numDevices = numDevices
        self.num_time_slots = num_time_slots
        self.N = numAPs + numDevices  # Total number of nodes
    def generate_network_parameters(self, minD, maxD, minE, maxE):
        """
        Generate network parameters with proper time-varying channels and
        persistent blockage events for RF and WOC.
        """
        # Initialize matrices
        distance_matrix = np.random.uniform(minD, maxD, size=(self.N, self.N))
        snr_db_matrix_WOC = np.zeros((self.N, self.N, self.num_time_slots))
        capacity_matrix_WOC = np.zeros((self.N, self.N, self.num_time_slots))
        snr_db_matrix_RF = np.zeros((self.N, self.N, self.num_time_slots))
        capacity_matrix_RF = np.zeros((self.N, self.N, self.num_time_slots))
        
        # Generate energy for devices (APs have infinite energy)
        EnergyTotal = np.random.uniform(minE, maxE, size=(self.N - self.numAPs))
        EnergyTotal = np.concatenate(([np.inf] * self.numAPs, EnergyTotal))
        
        snr_matrix = np.zeros((2, self.N, self.N, self.num_time_slots))
        capacity_matrix = np.zeros((2, self.N, self.N, self.num_time_slots))
        SendEnergy = np.zeros((2, self.N, self.N, self.num_time_slots))
        RecieveEnergy = np.zeros((2, self.N, self.N, self.num_time_slots))

        # Pre-generate blockage patterns
        blockage_RF = np.zeros((self.N, self.N, self.num_time_slots), dtype=bool)
        blockage_WOC = np.zeros((self.N, self.N, self.num_time_slots), dtype=bool)

        # --- Create persistent blockage sequences ---
        for i in range(self.N):
            for j in range(self.N):
                t = 0
                while t < self.num_time_slots:
                    # Random chance to start a blockage
                    if np.random.rand() < 0.05:  # ~5% chance to start blockage
                        # RF blockage (2–6 steps)
                        dur_rf = np.random.randint(2, 7)
                        blockage_RF[i, j, t:t+dur_rf] = True
                        # WOC blockage (3–8 steps)
                        dur_woc = np.random.randint(3, 9)
                        blockage_WOC[i, j, t:t+dur_woc] = True
                        t += max(dur_rf, dur_woc)  # skip forward by the longer blockage
                    else:
                        t += 1

        # --- Calculate SNR/capacity per link and time slot ---
        for i in range(self.N):
            for j in range(self.N):
                d = distance_matrix[i, j]
                for t in range(self.num_time_slots):
                    # ========== RF CHANNEL ==========
                    if (blockage_RF[i, j, t]) or (i == j) or (i < self.numAPs and j < self.numAPs):
                        snr_db_matrix_RF[i, j, t] = -20
                        capacity_matrix_RF[i, j, t] = 0
                        SendEnergy[0, i, j, t] = RecieveEnergy[0, i, j, t] = 1e50
                    else:
                        rician = self._generate_rician_fading(k_factor=10)
                        shadowing = np.random.normal(0, sigma_shadowing)
                        path_loss_RF = PL0 + 10 * attenuation_coefficient_RF * np.log10(d / d0) + shadowing
                        G_tx_linear = 10 ** (G_tx_RF / 10)
                        G_rx_linear = 10 ** (G_rx_RF / 10)
                        P_rx_RF = P_tx_RF * (np.sqrt(rician) ** 2) * 10 ** (-path_loss_RF / 10) * G_tx_linear * G_rx_linear
                        noise_power = k_B * Temperature * B_RF
                        snr_linear_RF = P_rx_RF / noise_power
                        snr_RF_db = 10 * np.log10(snr_linear_RF)
                        channel_capacity_RF = B_RF * np.log2(1 + snr_linear_RF)
                        snr_db_matrix_RF[i, j, t] = snr_RF_db
                        capacity_matrix_RF[i, j, t] = channel_capacity_RF
                        SendEnergy[0, i, j, t] = P_tx_RF / channel_capacity_RF
                        RecieveEnergy[0, i, j, t] = P_rx_RF / channel_capacity_RF

                    # ========== WOC CHANNEL ==========
                    if (blockage_WOC[i, j, t]) or (i == j) or \
                    (i < self.numAPs and j < self.numAPs) or (i >= self.numAPs and j >= self.numAPs):
                        snr_db_matrix_WOC[i, j, t] = -20
                        capacity_matrix_WOC[i, j, t] = 0
                        SendEnergy[1, i, j, t] = RecieveEnergy[1, i, j, t] = 1e50
                    else:
                        angle_variation = np.random.normal(0, 1.5)
                        theta_eff = theta + np.deg2rad(angle_variation)
                        phi_eff = phi + np.deg2rad(angle_variation)
                        P_signal = (P_tx_WOC * T_optical * R * IlluminationCoefficient *
                                ((u + 1) * A_rec * (np.cos(theta_eff) ** u) * np.cos(phi_eff)) /
                                (2 * np.pi * (d ** 2)))
                        I = R_photo * P_signal
                        In = R_photo * P_N
                        N_shot = np.sqrt(2 * q * (I + In) * B_WOC)
                        N_thermal = np.sqrt((4 * k_B * Temperature) / R_L * B_WOC)
                        N_total = np.sqrt(N_shot ** 2 + N_thermal ** 2)
                        snr_WOC = I ** 2 / N_total ** 2
                        snr_WOC_db = 10 * np.log10(snr_WOC)
                        channel_capacity_WOC = B_WOC * np.log2(1 + snr_WOC)
                        snr_db_matrix_WOC[i, j, t] = snr_WOC_db
                        capacity_matrix_WOC[i, j, t] = channel_capacity_WOC
                        SendEnergy[1, i, j, t] = (P_tx_WOC / R_photo) / channel_capacity_WOC
                        RecieveEnergy[1, i, j, t] = (I * V_bias) / channel_capacity_WOC

                    # Combine
                    snr_matrix[0, i, j, t] = snr_db_matrix_RF[i, j, t]
                    snr_matrix[1, i, j, t] = snr_db_matrix_WOC[i, j, t]
                    capacity_matrix[0, i, j, t] = capacity_matrix_RF[i, j, t]
                    capacity_matrix[1, i, j, t] = capacity_matrix_WOC[i, j, t]

        return (distance_matrix, snr_db_matrix_WOC, capacity_matrix_WOC,
                snr_db_matrix_RF, capacity_matrix_RF, snr_matrix, capacity_matrix,
                SendEnergy, RecieveEnergy, EnergyTotal)


    def _generate_rician_fading(self, k_factor=10):
        """Generate a new Rician fading sample each call"""
        sigma = 1 / np.sqrt(2 * (k_factor + 1))
        X = np.random.normal(k_factor * sigma, sigma)
        Y = np.random.normal(0, sigma)
        return np.sqrt(X**2 + Y**2)
    def generate_messages(self, minSize, maxSize, appSize, data_gen_prob):
        """
        Generate message queues with different priorities and sizes.
        
        Args:
            minSize (int): Minimum message size
            maxSize (int): Maximum message size
            appSize (int): Application-specific size
            data_gen_prob (float): Probability of message generation
            
        Returns:
            tuple: (T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp)
        """
        T = np.zeros((self.num_time_slots, self.N, self.N))
        MessageBegin = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        MessageEnd = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        msgQueues = [[[] for _ in range(self.N)] for _ in range(self.N)]
        msgApp = [[[] for _ in range(self.N)] for _ in range(self.N)]
        BiggestMsg = 0
        
        for i in range(self.N):
            for j in range(self.N):
                if i == j or (i < self.numAPs and j < self.numAPs):  # no messages between APs or self
                    continue
                    
                # Bernoulli process for each time slot
                for t in range(self.num_time_slots):
                    if np.random.random() < data_gen_prob:  # 70% probability of message
                        # Generate message length
                        msg_length = np.random.randint(minSize, maxSize + 1)
                        if t + msg_length > self.num_time_slots:
                            continue
                            
                        # Create message
                        msg_id = len(msgQueues[i][j])
                        msgQueues[i][j].append(list(range(t, t + msg_length)))
                        MessageBegin[i][j][msg_id] = t
                        MessageEnd[i][j][msg_id] = t + msg_length
                        
                        # Update T matrix
                        for k in range(t, t + msg_length):
                            T[k][i][j] = 1
                            
                        # Update BiggestMsg
                        if msg_length > BiggestMsg:
                            BiggestMsg = msg_length
                            
                        # Assign application type
                        app_type = np.random.randint(1, appSize + 1)
                        msgApp[i][j].append([app_type] * msg_length)
        
        return T, MessageBegin, MessageEnd, msgQueues, BiggestMsg, msgApp
                            
    def generate_size(self, msgQueues, sizemin, sizemax, S_p):
        """
        Generate message sizes for the queues.
        
        Args:
            msgQueues (list): List of message queues
            sizemin (int): Minimum size
            sizemax (int): Maximum size
            S_p (int): Packet size
            
        Returns:
            tuple: (Pt, Pt_number, maxN)
        """
        #low, meduim, high
        Pt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        Pt_number = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        maxN=0
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                    for nmessage in range(0,len(msgQueues[i][j])):
                        Pt[i][j][nmessage]=random.randint(sizemin, sizemax) # generate between 1 and 20 Megabites
                        Pt_number[i][j][nmessage]=int(Pt[i][j][nmessage]/S_p) # generate between 1 and 20 Megabites
                        maxN= int(Pt[i][j][nmessage]/S_p) if int(Pt[i][j][nmessage]/S_p)>maxN else maxN

                        
                            
                        
        return Pt,Pt_number,maxN
        

