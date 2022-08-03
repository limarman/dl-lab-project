# DL Lab Project - Reinforcement Learning
This repository contains the code base for the Kore Game Project written by Daniel, Ramil, Nils and Hasham.  

## Project Structure 
The project is divided into a jobscripts, src and test folder. Furthermore, there are files for the automatic unit testing and linting with GitHub actions. The test folder contains selected unit tests for important functions and the jobscripts folder contains one example script with comments on how to ajust it properly. The src folder has the following structure:
* Actions
  - Rule based Actions
  - Action Adapters
  - Validity Action Masking
* Agents
  - A2C, DQN 
  - Neural Networks (Pytorch for A2C/PPO, Keras for DQN)
  - Baselines / Opponent Agents
  - Policies for DQN
  - Loggers for Training (e.g. wandb)
* Environment
  - Gym environment (substep) for the Kore Game
  - Environment Factory
* Monitoring
  - Monitor for the Environment
* Rewards
  - Different Reward Functions
* States
  - Multiple States (e.g. for Hybrid Networks and the multi-modal transformer)
  - BoardWrapper for computing input features
* core
  - Example for training and evaluating agents
* experiments
  - Example experiments and utility functions





## Running experiments on the RWTH cluster
Login `ssh -l username login18-1.hpc.itc.rwth-aachen.de`

Configure the environemnt variables for weights and biases, i.e. execute the following commands at the cluster:
`WANDB_API_KEY=YOUR_API_KEY` `WANDB_ENTITY=rl-dl-lab` `export WANDB_API_KEY, WANDB_ENTITY` 

Transfer code to the cluster, e.g.
`rsync -a -e ssh --exclude output --exclude venv dl-lab-project username@copy.hpc.itc.rwth-aachen.de:lab`


Activate your virtual environment and install new requirements (use intel optimized distributions) `conda activate lab-env` `conda install --file requirements.txt`

Perform small test run `python3 -m src.core.main`

Make a suitable jobscript and submit it (e.g. `sbatch jobscript.sh`) for a short time (e.g. a minute) to check if everything works. After that adjust the required time and resources before resubmitting it. You may submit chain jobs. The training is automatically saved and reloaded in the src/core/main.py example. Furthermore, you may adjust the timeframe for the TERM interrupt to run evaluations after training.


## Acknowledgements
We utilized many reinforcement learning frameworks such stablebaselines3 and keras-rl-2 as well as the deep learning frameworks PyTorch and Keras to rapidly implement the agents and to rely on well-established software packages. See the requirements.txt file for a complete overview of dependecies. In the code, we give additional credit whenever we have adapted code from others (e.g. from tutorials).
