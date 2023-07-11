#!/bin/bash
#SBATCH -N 1                       # Request 1 node
#SBATCH -c 28                      # Request 28 cores (assuming each long-28core node has 28 cores)
#SBATCH -t 48:00:00                # Set maximum runtime to 48 hours
#SBATCH -p long-28core             # Use the long cores

# Load any necessary modules or activate your virtual environment (if required for predictability.py)
module load anaconda
source activate myenv

# Run the predictability.py file
python predictability.py
