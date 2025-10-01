#!/bin/bash
#SBATCH --job-name=f61n1000sdbench      # Job name
#SBATCH --output=f61n1000sdbench.log        # Output log file
#SBATCH --error=errorf61n1000sdbench.log          # Error log file
#SBATCH --mail-type=END            # Send email at job end
#SBATCH --mail-user=amachas@materials.uoc.gr  # Email address for notifications

# Activate a virtual environment (if needed)
source ~/venv/bin/activate
# Run the Python script
jfsd -c f61n1000sdbench/input.toml -s f61n1000sdbench/f61n1000after10000nooverlaps.npy -o f61n1000sdbench/
