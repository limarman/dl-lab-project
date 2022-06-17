# DL Lab Project - Reinforcement Learning



## Running experiments on the RWTH cluster
Login `ssh -l username login18-1.hpc.itc.rwth-aachen.de`

Configure the environemnt variables for weights and biases, i.e. execute the following commands at the cluster:
`WANDB_API_KEY=YOUR_API_KEY` `WANDB_ENTITY=rl-dl-lab` `export WANDB_API_KEY, WANDB_ENTITY` 

Transfer code to the cluster, e.g.
`rsync -a -e ssh --exclude output --exclude venv dl-lab-project username@copy.hpc.itc.rwth-aachen.de:lab`


Activate your virtual environment and install new requirements (use intel optimized distributions) `conda activate lab-env` `conda install --file requirements.txt`

Perform small test run `python3 -m src.core.main`

Make a suitable jobscript and submit it (e.g. `sbatch jobscript.sh`) for a short time (e.g. a minute) to check if everything works. After that adjust the required time and resources before resubmitting it.


