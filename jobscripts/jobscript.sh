#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1
#SBATCH --array=1-2%1

#SBATCH --cpus-per-task=24

# ask for less than 4 GB memory per task=MPI rank
#SBATCH --mem-per-cpu=1900M   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

# name the job
#SBATCH --job-name=CLUSTER_TEST

# declare the merged STDOUT/STDERR file
#SBATCH --output=/home/ob606396/lab/dl-lab-project/jobscripts/output.%J.txt

# setting time to one minute (--time=d-hh:mm:ss)
#SBATCH --time=0-00:05:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=nils.eberhardt@rwth-aachen.de

#SBATCH --account=lect0082


# Change to working directory
cd /home/ob606396/lab/dl-lab-project

# make conda available
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

export WANDB_ENTITY=rl-dl-lab

# activate environement
conda activate lab3.10

# beginning of executable commands
python3 -m src.core.main
