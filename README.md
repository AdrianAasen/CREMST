# Correlated readout error mitigated state tomography (CREMST)

This codebase builds upon the REMST codebase found on [this repository](https://github.com/AdrianAasen/EMQST) to perform correlated readout error mitigation for quantum systems.

## Publications

The following publications used this protocol:\
[1] [Readout error mitigated quantum state tomography tested on superconducting qubits](https://www.nature.com/articles/s42005-024-01790-8) Communications Physics volume **7**, Article number: 301 (2024) (uses older version of the codebase).\
[2] [Multiplexed qubit readout quality metric beyond assignment fidelity](https://journals.aps.org/pra/abstract/10.1103/6p6s-t8b7)  Phys. Rev. A **112**, 022601 (2025).\
[3] [Mitigation of correlated readout errors without randomized measurements](https://arxiv.org/abs/2503.24276) (2025 preprint).

This repository contains code, results, and example notebooks developed for each of the publication. Each major project or paper has its own folder under `Paper_results_and_notebooks`, containing:
- **Code** for simulations and data analysis
- **Results** (numerical and visual outputs)
- **Example notebooks** for reproducing figures and workflows

---

#### ▸ `Mitigation_of_correlated_readout_errors/`
- `cluster_code/` – Batch/cluster execution code.  
- `Correlated_QREM_results/` – Results of correlated readout error mitigation.  
- `Exp_povms/` – POVM experiment data.  
- `images/` – Figures and plots for the paper.  
- `paper_visualization.ipynb` – Notebook for reproducing paper figures.  
- `Scalable_QREM.ipynb` – Example scalable QREM notebook.  

#### ▸ `Multiplexed_qubit_readout_quality_metric/`
- `Example_experimental_povms.ipynb` – Example notebook on how to load in the experimental POVMs characterized. 
- `Multiplexed_qubit_readout_visualization.ipynb` – Notebook for reproducing paper figures and additional simulations
- The rest is figure and result folders. 


#### ▸ `Readout_error_mitigated_quantum_state_tomography/`
- `Tutorial.ipynb` – Example notebook on how to perform BME and how to extract properties, and how it can be used to perform readout error mitigation.
