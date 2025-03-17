# Correlated readout error mitigated state tomography (CREMST)

This codebase builds upon the REMST codebase found on [this repository](https://github.com/AdrianAasen/EMQST) to perform correlated readout error mitigation for quantum systems.

## Publications

The following publications used this protocol:
[1] [Readout error mitigated quantum state tomography tested on superconducting qubits](https://www.nature.com/articles/s42005-024-01790-8) (2024) (uses older version of the codebase).\
[2] [Multiplexed qubit readout quality metric beyond assignment fidelity](https://arxiv.org/abs/2502.08589) (2025 preprint).\
[3] Mitigation of correlated readout errors without randomized measurements (in preparation).\

## Code Structure

- **Notebook paper_visualization**: Includes example code to display the results used in ref. [3].
- **cluster_code**: Contains the code blocks used to run the simulations in ref. [3] on a standard SLURM cluster.
- **initializers scripts**: Used to create an initializer object that runs multiple simulations with identical initial conditions.

## Tutorial

A user-friendly tutorial notebook is provided that shows how to run CREMST with examples of how to create your own simulations and load data from previous runs. 

## Examples

There are examples of other applications, such as:
- Extracting specific basis states from detector tomography.
- Computing the standard deviation of the BME infidelities.
