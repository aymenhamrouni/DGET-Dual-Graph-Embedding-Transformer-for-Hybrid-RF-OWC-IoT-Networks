# DGET: Dual-Graph Embedding Transformer for Hybrid RF–OWC IoT Networks

## Overview
This repository provides the complete, reproducible implementation accompanying the paper:  
**“Resource Allocation in Hybrid Radio-Optical IoT Networks using GNN with Multi-task Learning.”**

It implements the **Dual-Graph Embedding with Transformer (DGET)** framework for **hybrid Radio Frequency (RF)** and **Optical Wireless Communication (OWC)** IoT networks.  
The pipeline integrates:

- A **Mixed-Integer Nonlinear Programming (MINLP)** optimization module to generate ground-truth scheduling data.  
- A **multi-task deep learning model** combining transductive and inductive Graph Neural Networks (GNNs) with a Transformer encoder for temporal and cross-link prediction.

---

## System Pipeline

### Step 1 – Optimization and Dataset Generation
- **Scripts:** `main.py`, `utils.py`, `MINLP_model.py`
- **Purpose:** Solves a **bi-objective MINLP** to jointly maximize throughput and minimize delay under device energy and link availability constraints.  
- Generates **input (`GI`)** and **recorded (`GR`)** temporal graph snapshots representing device energy, queue states, and hybrid RF–OWC link evolution.  
- Stores labeled CSV datasets in the `dataset/` directory.

---

### Step 2 – Dual Graph Embedding and Model Training
- **Scripts:** `train_DGET.py`, `utils_DGET.py`
- **Architecture:**
  - **Transductive GNN (Link-feature-aware GAT):**  
    Learns structural and relational embeddings from fully observed graphs, weighting neighbors by link type and importance.
  - **Inductive GNN (Edge-enhanced GraphSAGE):**  
    Generalizes transductive embeddings to evolving graph states by aggregating decision-level and temporal information.  
    Trained via a **consistency loss** aligning transductive and recorded embeddings.
  - **Transformer Encoder:**  
    Applies multi-head self-attention across temporal embeddings to model long-range dependencies and predict per-link communication classes.

---

### Step 3 – Evaluation and Visualization
- Automatically generates:
  - ROC / AUC curves  
  - Confusion matrices (clean and aged data)  
  - Accuracy / loss curves  
  - Energy, AoI, and throughput plots  
- Performs **post-processing correction** via a **top-2 feasibility check** ensuring valid link allocations.  
- Saves models and plots in `/models` and `/output`.

---

## Model Summary

- **Multi-task learning objective:**
  - *Classification loss* — cross-entropy with class weighting  
  - *Consistency loss* — MSE alignment of transductive and recorded embeddings  
- **Augmented edge labeling** and **weighted penalties** mitigate dataset imbalance.  
- **Loss:**  
  \[
  \mathcal{L} = \mathcal{L}_{classification} + \lambda \mathcal{L}_{consistency}
  \]

---

## Model Architecture

![DGET Architecture](figures/dget_architecture.png)
*Figure 1 – High-level architecture of the Dual-Graph Embedding Transformer (DGET) framework.*


## Key Features
- End-to-end **hybrid RF–OWC** simulation and graph-learning pipeline  
- **Dual-stage embedding** (Transductive + Inductive GNN)  
- **Transformer-based temporal link classifier**  
- **Feasibility-aware post-processing correction**  
- **Near-optimal scheduling** with significantly reduced complexity versus MINLP  

---

## Dependencies

Install dependencies:
```bash
pip install -r requirements.txt


```
## Project Structure
```
.
├── DGET_train.py
├── DGET_utils.py
├── MINLP_model.py
├── constants.py
├── data_generation.py
├── main.py
├── utils.py
├── requirements.txt
├── .gitignore
└── dataset/               # Generated at runtime 
└── models/               # Generated at DGET training 
```
## Configuration Reference
- `main.py`
  - Network: `N_APs`, `N_d`, `Time`, `S_p`, `N_montecarlo`
  - Channel thresholds: `snr_min_rf`, `snr_min_woc`
  - Visualization/verbosity: `VIZ`, `Verbose`
  - Workload: `start`, `end`, `step` (packet sizes)
  - Sim suites: calls `run_APs_simulation` and `run_Nd_simulation`
- `data_generation.py`
  - Draws distances and time-varying SNRs, capacities, energies, message queues
- `MINLP_model.py`
  - DOcplex model with variables: technology selection `x`, message selection `x_m`, status, delays `delta`, unsent counts, and constraints
- `utils.py`
  - Builds CSVs (both input and recorded), computes AoI metrics (per pair and global), energy/throughput plots, and switching analysis
- `DGET_train.py`
  - Set `size` to dataset `N` before running
  - Model: temporal encoder + GAT + (weighted) GraphSAGE + MLP edge classifier
  - Training controls: `epochs`, `n_repeats` (for repeated K-fold), OneCycle LR scheduler, class-weighted loss

  Generated files (columns are produced by the code):
- `dataset/inputNodesFeaturesN.csv`
  - `TimeStep, NodeID, EnergyLevel`
- `dataset/inputEdgesFeaturesN.csv`
  - `TimeStep, SourceNode, TargetNode, SendingStatus, RF_SendEnergy, RF_ReceiveEnergy, WOC_SendEnergy, WOC_ReceiveEnergy, MessageQueueLength, AllowedTechnology`
  - `AllowedTechnology` encodes feasibility: 0=none, 1=RF, 2=WOC, 3=both
- `dataset/recordedNodesFeaturesN.csv`
  - Time-series of energy after applying schedules (`EnergyLevel` per node/time)
- `dataset/recordedEdgesFeaturesN.csv`
  - `TimeStep, SourceNode, TargetNode, SendingStatus, QueueLength, AllowedTechnology, SelectedTechnology, Label`
  - `SelectedTechnology` is 0 (none), 1 (RF), 2 (WOC)
  - `Label` derives from feasibility/selection combinations and is used as the training target


Notes:
- `docplex` installs from pip. For large scenarios or exact solving, install IBM CPLEX and set environment variables; otherwise small/medium runs can proceed with community behavior.
- Backend for Matplotlib is set to `TkAgg` in `main.py` and `DGET.py`. Make sure Tk is available or switch the backend if needed.


## Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/awesome-feature`
3. Commit changes with clear messages: `git commit -m "feat: add awesome feature"`
4. Push the branch: `git push origin feature/awesome-feature`
5. Open a Pull Request describing the motivation, approach, and tests/plots if relevant


## Contact
- Aymen Hamrouni: aymen.hamrouni@kuleuven.be
- Sofie Pollin: sofie.pollin@kuleuven.be
- Hazem Sallouha: hazem.sallouha@kuleuven.be