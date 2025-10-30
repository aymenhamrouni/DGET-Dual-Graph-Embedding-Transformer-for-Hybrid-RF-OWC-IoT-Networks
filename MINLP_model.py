"""
Mixed Integer Non-Linear Programming Model for Hybrid RF-WOC Network Optimization

This module implements a MINLP model for optimizing communication in hybrid RF-WOC networks.
The model considers multiple objectives and constraints:

1. Objectives:
   - Maximize message delivery
   - Minimize energy consumption
   - Maximize network capacity
   - Minimize transmission delay
   - Minimize technology switching
   - Multi-objective optimization

2. Constraints:
   - Network topology
   - Energy limitations
   - Signal quality requirements
   - Message scheduling
   - Technology switching

3. Key Features:
   - Support for multiple communication modes
   - Age of Information (AoI) metrics
   - Energy-aware scheduling
   - Context-aware optimization

Author: Aymen Hamrouni
Date: 2024
"""

import docplex.mp.model as cpx
from collections import defaultdict

def MINLP_model(M_OWC, M, maxN, N, Time, msgQueues, BiggestMsg, MsgNumber, T, 
                snr_db_matrix_RF, snr_db_matrix_OWC, snr_min_rf, snr_min_owc,
                capacity_matrix, S_p, Time_unit, SendEnergy, RecieveEnergy, EnergyTotal,
                objective_type='max_messages', Timeout=60):
    """
    Formulate and solve the MINLP model for hybrid RF-WOC network optimization.
    
    This function implements the core optimization model with the following components:
    1. Decision Variables:
       - Communication mode selection
       - Message transmission scheduling
       - Node status tracking
       - Delay variables
    
    2. Constraints:
       - Single mode communication
       - Energy limitations
       - Signal quality requirements
       - Message scheduling
       - Technology switching
    
    3. Objectives:
       - Message maximization
       - Energy minimization
       - Capacity maximization
       - Delay minimization
       - Switching minimization
       - Multi-objective optimization
    
    Args:
        M_OWC (int): Starting mode index for OWC
        M (int): Total number of communication modes
        maxN (int): Maximum number of messages
        N (int): Number of nodes
        Time (int): Number of time slots
        msgQueues (list): Message queues for each node pair
        BiggestMsg (int): Maximum message size
        MsgNumber (list): Number of messages per node pair
        T (list): Topology matrix
        snr_db_matrix_RF (numpy.ndarray): RF SNR matrix
        snr_db_matrix_OWC (numpy.ndarray): OWC SNR matrix
        snr_min_rf (float): Minimum RF SNR
        snr_min_owc (float): Minimum OWC SNR
        capacity_matrix (numpy.ndarray): Channel capacity matrix
        S_p (int): Packet size
        Time_unit (float): Time unit duration
        SendEnergy (numpy.ndarray): Energy for sending
        RecieveEnergy (numpy.ndarray): Energy for receiving
        EnergyTotal (list): Total available energy
        objective_type (str): Optimization objective type
        Timeout (int): Solver timeout
    
    Returns:
        tuple: (MINLP_solution, opt_model, x, x_m, status, delta, Unsent)
    """
    Max1 = 0  # max value for subobjective 1
    Max2 = 0  # max value for subobjective 2
    Max3 = 0  # max value for subobjective 3
    Max4= 0  # max value for subobjective 4

    # Create optimization model
    opt_model = cpx.Model(name="MINLP Hybrid RF-OWC Model")
    
    # Decision variables
    x = {(i, j, m, k): opt_model.binary_var(name="x_{}_{}_{}_{}".format(i, j, m, k))
         for i in range(N) for j in range(N) for m in range(M_OWC,M) for k in range(Time)}
    
    # Message transmission variables
    x_m = {}
    for i in range(N):
        for j in range(N):
            for f in range(0, len(msgQueues[i][j])):
                x_m.update({(i, j, f): opt_model.binary_var(
                    name="xm_{}_{}_{}".format(i, j, f))})
    
    # Delay variables
    delta = {(i, j, f): opt_model.integer_var(lb=0, ub=BiggestMsg+1, name="delta_{}_{}_{}".format(i, j, f))
             for i in range(N) for j in range(N) for f in range(0, len(msgQueues[i][j]))}
    
    # Unsent messages variables
    Unsent = {(i, j, f): opt_model.integer_var(lb=0, ub=maxN, name="Unsent_{}_{}_{}".format(i, j, f))
             for i in range(N) for j in range(N) for f in range(0, len(msgQueues[i][j]))}
    
    # Node status variables
    status = {(i, k): opt_model.binary_var(
        name="status_{}_{}".format(i, k)) for i in range(N) for k in range(Time)}
    
    # Temporary variables for constraints
    temp0 = {(i, j, k): opt_model.binary_var(name="temp0_{}_{}_{}".format(
        i, j, k)) for i in range(N) for j in range(N) for k in range(Time)}
    temp1 = {(i, j, k): opt_model.binary_var(name="temp1_{}_{}_{}".format(
        i, j, k)) for i in range(N) for j in range(N) for k in range(Time)}
    
    # Constraints
    # 1. Node can choose only one mode and one node to communicate with at time step k
    constraints1 = {(i, k): opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for j in range(
        N) for m in range(M_OWC,M)) <= 1, ctname="constraints1_{}_{}".format(i, k)) for i in range(N) for k in range(Time)}
    
    # 2. Node can only receive from one node with one mode at time step k
    constraints2 = {(j, k): opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for i in range(
        N) for m in range(M_OWC,M)) <= 1, ctname="constraints2_{}_{}".format(j, k)) for j in range(N) for k in range(Time)}
    
    # 3. Communication must be allowed by topology
    constraints3 = {(i, j, m, k): opt_model.add_constraint(ct=T[k][i][j] >= x[i, j, m, k], ctname="constraints3_{}_{}_{}_{}".format(
        i, j, m, k)) for i in range(N) for j in range(N) for m in range(M_OWC,M) for k in range(Time)}
    
   
    # 6. No self-communication
    constraints4 = {(i, m, k): opt_model.add_constraint(ct=x[i, i, m, k] == 0, ctname="constraints4_{}_{}_{}".format(
        i, m, k)) for i in range(N) for m in range(M_OWC,M) for k in range(Time)}
    
    # 7. No simultaneous sending and receiving
    constraints5 = {(i, j, m, m2, f, k): opt_model.add_constraint(ct=x[i, j, m, k]+x[j, f, m2, k] <= 1, ctname="constraint5_{}_{}_{}_{}_{}_{}".format(
        i, j, m, m2, f, k)) for i in range(N) for j in range(N) for m in range(M_OWC,M) for m2 in range(M_OWC,M) for f in range(N) for k in range(Time)}
    
    # Message transmission constraints
    z_m = {}
    for i in range(N):
        for j in range(N):
            opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for m in range(M_OWC,M) for k in set([item for sublist in msgQueues[i][j] for item in sublist])) == opt_model.sum(
                x_m[i, j, nmessage] for nmessage in range(0, len(msgQueues[i][j]))), ctname="constraints6_{}_{}".format(i, j))
    
            for nmessage in range(0, len(msgQueues[i][j])):
                subrange = list(msgQueues[i][j][nmessage])
                
                # Message transmission constraints
                opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for m in range(M_OWC,M) for k in subrange) <= 1, ctname="constraints7_{}_{}".format(i, j))
                opt_model.add_constraint(ct=opt_model.sum(x[i, j, m, k] for m in range(M_OWC,M) for k in subrange) >= x_m[i, j, nmessage], ctname="constraints8_{}_{}".format(i, j))
                opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, k] for m in range(M_OWC,M) for k in subrange) == 1, x_m[i, j, nmessage] == 1), ctname="constraints9_{}_{}".format(i, j))
                opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, k] for m in range(M_OWC,M) for k in subrange) == 0, x_m[i, j, nmessage] == 0), ctname="constraints10_{}_{}".format(i, j))
                opt_model.add_constraint(ct=opt_model.if_then(x_m[i, j, nmessage] == 1, opt_model.sum(
                    x[i, j, m, k] for m in range(M_OWC,M) for k in subrange) == 1), ctname="constraints11_{}_{}".format(i, j))
                
                # Additional variables and constraints for message transmission
                z = opt_model.binary_var_dict(
                    [(i, j, nmessage, sub) for sub in subrange],
                    name="z"
                )

                opt_model.add_constraint(
                    ct=opt_model.sum(z[i, j, nmessage, sub] for sub in subrange) <= 1,
                    ctname="mutual_exclusivity_{}_{}".format(i, j)
                )
                
                for sub in subrange:
                    z_m.update({(i, j, 0, nmessage, sub): opt_model.integer_var(
                        name="zm_{}_{}_{}_{}_{}".format(i, j, 0, nmessage, sub))})
                    z_m.update({(i, j, 1, nmessage, sub): opt_model.integer_var(
                        name="zm_{}_{}_{}_{}_{}".format(i, j, 1, nmessage, sub))})
                    
                    # Message transmission constraints
                    opt_model.add_constraint(opt_model.sum(z_m[i, j, m, nmessage, sub] for m in range(M_OWC,M)) <= maxN * opt_model.sum(x[i, j, m, sub] for m in range(M_OWC,M)),
                         ctname="z_upper_bound1_{}_{}_{}_{}".format(i, j, nmessage, sub))
                    opt_model.add_constraint(opt_model.sum(z_m[i, j, m, nmessage, sub] for m in range(M_OWC,M)) <= Unsent[i,j,nmessage],
                         ctname="z_upper_bound2_{}_{}_{}_{}".format(i, j, nmessage, sub))
                    opt_model.add_constraint(opt_model.sum(z_m[i, j, m, nmessage, sub] for m in range(M_OWC,M)) >= 0, ctname="z_lower_bound1_{}_{}_{}_{}".format(i, j, nmessage, sub))
                    opt_model.add_constraint(opt_model.sum(z_m[i, j, m, nmessage, sub] for m in range(M_OWC,M)) >= (Unsent[i,j,nmessage]) - (1 - opt_model.sum(x[i, j, m, sub] for m in range(M_OWC,M)))*maxN,
                                          ctname="z_lower_bound2_{}_{}_{}_{}".format(i, j, nmessage, sub))
                    
                    opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(z_m[i, j, m, nmessage, sub] for m in range(M_OWC,M))==maxN, opt_model.sum(x[i, j, m, sub] for m in range(M_OWC,M))==0), 
                                          ctname="linearizationConstraints_{}_{}_{}_{}".format(i, j, nmessage, sub))

                    # Link z to x variables
                    opt_model.add_constraint(
                        ct=z[i, j, nmessage, sub] == opt_model.sum(x[i, j, m, sub] for m in range(M_OWC,M)),
                        ctname="link_z_x_{}_{}_{}_{}".format(i, j, nmessage, sub))
                    
                    # Delta constraints
                    opt_model.add_constraint(
                        ct=delta[i, j, nmessage] == opt_model.sum(
                            z[i, j, nmessage, sub] * ((sub - subrange[0]) + 1) for sub in subrange
                        ) + (1 - opt_model.sum(z[i, j, nmessage, sub] for sub in subrange)) * (BiggestMsg + 1),
                        ctname="unique_delta_{}_{}".format(i, j))
                    
                    # Capacity constraints
                    if M_OWC == 0:
                        opt_model.add_constraint(ct=0 <= x[i, j, 0, sub]* MsgNumber[i][j][nmessage] - z_m[i,j,0,nmessage,sub], 
                                              ctname="constraint12_{}_{}".format(i,j))
                        opt_model.add_constraint(ct=x[i, j, 0, sub]* MsgNumber[i][j][nmessage] - z_m[i,j,0,nmessage,sub] <= (capacity_matrix[0][i][j][sub]/S_p)*Time_unit, 
                                              ctname="constraint13_{}_{}".format(i,j))
                    if M > 1:
                        opt_model.add_constraint(ct=x[i, j, 1, sub]* MsgNumber[i][j][nmessage] - z_m[i,j,1,nmessage,sub] <= (capacity_matrix[1][i][j][sub]/S_p)*Time_unit, 
                                              ctname="constraint14_{}_{}".format(i,j))
                        opt_model.add_constraint(ct=0 <= x[i, j, 1, sub]* MsgNumber[i][j][nmessage] - z_m[i,j,1,nmessage,sub], 
                                              ctname="constraint15_{}_{}".format(i,j))
                
                Max3 = Max3 + BiggestMsg + 1
    
    # Node status constraints
    constraints16 = {(j, m, k): opt_model.add_constraint(ct=opt_model.if_then((opt_model.sum(x[i, j, m, k] for i in range(N)) + opt_model.sum(
        x[j, i, m, k] for i in range(N))) == 1, status[j, k] == m), ctname="constraint16_{}_{}_{}".format(j, m, k)) for j in range(N) for m in range(M_OWC,M) for k in range(Time)}
    
    # Status transition constraints
    constraints17 = {(j, k): opt_model.add_constraint(ct=opt_model.logical_and((opt_model.sum(opt_model.sum(x[i, j, m, k]+x[j, i, m, k] for i in range(N)) for m in range(M_OWC,M))) == 0, status[j, k-1] == 0) <= (status[j, k] == 0), 
                     ctname="constraint17_{}_{}".format(j, k)) for j in range(N) for k in range(1, Time)}
    
  

    constraints18 = {(j, k): opt_model.add_constraint(ct=opt_model.logical_and((opt_model.sum(opt_model.sum(x[i, j, m, k]+x[j, i, m, k] for i in range(N)) for m in range(M_OWC,M))) == 0, status[j, k-1] == 1) <= (status[j, k] == 1), 
                     ctname="constraint18_{}_{}".format(j, k)) for j in range(N) for k in range(1, Time)}
    
    # Energy constraints
    constraints19 = {(i): opt_model.add_constraint(ct=opt_model.sum((x[i, j, m, k]* MsgNumber[i][j][nmessage] - z_m[i,j,m,nmessage,k])*(SendEnergy[m][i][j][k]+RecieveEnergy[m][i][j][k])*S_p for j in range(
        N) for m in range(M_OWC,M) for nmessage in range(0, len(msgQueues[i][j])) for k in list(msgQueues[i][j][nmessage])) <= EnergyTotal[i], ctname="constraint7_{}".format(i)) for i in range(N)}
    
    # Initial status constraints
    constraints20 = {(j): opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, m, 0]+x[j, i, m, 0] for i in range(N) for m in range(M_OWC,M)) == 0, status[j, 0] == 0), 
                    ctname="constraint20_{}".format(j)) for j in range(N)}
    
     
    

    if M_OWC == 0:
        constraints21  = {(j): opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, 0, 0]+x[j, i, 0, 0] for i in range(N)) == 1, status[j, 0] == 0), 
                         ctname="constraint21_{}".format(j)) for j in range(N)}
         # 4. RF communication feasibility
        constraints22 = {(i, j, k): opt_model.add_constraint(ct=temp0[i, j, k] * snr_db_matrix_RF[i][j][k] >= snr_min_rf*x[i, j, 0, k],
                                                            ctname="constraint22_{}_{}_{}".format(i, j, k)) for i in range(N) for j in range(N) for k in range(Time)}
    
    if M > 1:
        # 5. OWC communication feasibility
        constraints23 = {(i, j, k): opt_model.add_constraint(
            ct=temp1[i, j, k] * snr_db_matrix_OWC[i][j][k] >= snr_min_owc*x[i, j, 1, k], ctname="constraint23_{}_{}_{}".format(i, j, k)) for i in range(N) for j in range(N) for k in range(Time)}
   
        # No more than 50% switching 
        #constraints24 = {opt_model.add_constraint(ct=opt_model.sum(opt_model.abs(status[j, k]-status[j, k-1]) for j in range(0, N) for k in range(1, Time)) <= Time//4, ctname="constraints24")}
       # Initial technology constraints 
        constraints25 = {(j): opt_model.add_constraint(ct=opt_model.if_then(opt_model.sum(x[i, j, 1, 0]+x[j, i, 1, 0] for i in range(N)) == 1, status[j, 0] == 1), 
                         ctname="constraint25{}".format(j)) for j in range(N)}
    
    # Calculate maximum values for normalization

    for i in range(N):
        for j in range(N): 
            for k in range(Time):
                if snr_db_matrix_RF[i][j][k] >= snr_min_rf and T[k][i][j] >= 1:
                    Max1 = Max1 + (SendEnergy[0][i][j][k] + RecieveEnergy[0][i][j][k])
                if snr_db_matrix_OWC[i][j][k] >= snr_min_owc and T[k][i][j] >= 1 and M > 1:
                    Max1 = Max1 + (SendEnergy[1][i][j][k] + RecieveEnergy[1][i][j][k])
    
    Max0 = sum(MsgNumber[i][j][nmessage] for i in range(len(MsgNumber)) 
               for j in range(len(MsgNumber[i])) 
               for nmessage in range(len(MsgNumber[i][j])))*M
    Max2 = sum(1 for j in range(0, N) for k in range(1, Time))
    Max3 = sum(BiggestMsg+1 for j in range(0, N) for i in range(0, N) for k in range(0, Time))/(((N*(N-1))/2)/2) 
    Max4= sum(capacity_matrix[m][i][j][k] for i in range(N) for j in range(N) for m in range(M_OWC,M) for k in range(Time))/(((N*(N-1))/4)) # 1 tech possible, divide by number of possible relations 

    #Objective functions

    #Maximize capacity
    Obj0=(opt_model.sum( (x[i, j, m, k])*capacity_matrix[m][i][j][k] for m in range(M_OWC,M)  for i in range(N) for j in range(N) for k in range(Time)) )/Max4 

    #Minimize energy
    Obj1=(opt_model.sum(x[i, j, m, k]*T[k][i][j]*(SendEnergy[m][i][j][k]+RecieveEnergy[m][i][j][k]) for m in range(M_OWC,M) for i in range(N) for j in range(N) for k in range(Time)))/(Max1/2) 
    #Maximize number of messages 
    Obj2=(opt_model.sum(((x[i, j, m, k]*MsgNumber[i][j][nmessage])-z_m[i,j,m,nmessage,k]) 
                            for m in range(M_OWC,M) 
                            for i in range(N) 
                            for j in range(N) 
                            for nmessage in range(0, len(msgQueues[i][j])) 
                            for k in list(msgQueues[i][j][nmessage])))/(Max0)
    #Minimize number of messages w.r.t energy
    Obj3=(opt_model.sum(((x[i, j, m, k]*MsgNumber[i][j][nmessage])-z_m[i,j,m,nmessage,k])/(SendEnergy[m][i][j][k]+RecieveEnergy[m][i][j][k]) for m in range(M_OWC,M) for i in range(N) for j in range(N) for nmessage in range(0, len(msgQueues[i][j])) for k in list(msgQueues[i][j][nmessage])))

    #Minimize number of switching
    Obj4=-opt_model.sum(opt_model.abs(status[j, k]-status[j, k-1]) for j in range(0, N) for k in range(1, Time))/Max2
    #Minimize delay
    Obj5=opt_model.sum_squares(delta[i, j, f] for i in range(N) for j in range(N) for f in range(0, len(msgQueues[i][j])) )/((Max3*((N*(N-1))/2)))
    
    alpha1=0.7
    alpha2=0.3
    objective = alpha1*Obj2 -  alpha2*Obj1 + alpha2
   
    
    # Configure solver
    opt_model.context.cplex_parameters.optimalitytarget = 3  # Allow non-convex quadratic objectives
    opt_model.set_time_limit(Timeout)  # Set timeout from parameter
    
    # Select objective based on objective_type
    if objective_type == 'max_messages':
        opt_model.maximize(Obj2)
    elif objective_type == 'min_energy':
        opt_model.minimize(Obj1)
    elif objective_type == 'max_capacity':
        opt_model.maximize(Obj0)
    elif objective_type == 'min_delay':
        opt_model.minimize(Obj5)
    elif objective_type == 'min_switching':
        opt_model.minimize(Obj4)
    elif objective_type == 'max_messages-w.r.t.energy':
        opt_model.minimize(Obj3)
    elif objective_type == 'multi-objective-max-messages and min-energy':
        opt_model.maximize(objective)
    else:
        raise ValueError(f"Invalid objective_type: {objective_type}. Must be one of: 'max_messages', 'min_energy', 'max_capacity', 'min_delay', 'min_switching', 'max_messages-w.r.t.energy', 'multi-objective-max-messages and min-energy'")
    
    # Solve the model
    print(f'Solving Optimization Model with objective: {objective_type}...')
    MINLP_solution = opt_model.solve()
    print('Optimization Model Solved!')
    return MINLP_solution, opt_model, x, x_m, status, delta, Unsent 