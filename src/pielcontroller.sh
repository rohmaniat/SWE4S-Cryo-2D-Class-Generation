#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --output=download_data.log
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

# Optional: load any required modules (depends on your cluster)
# module load curl

# Create a data directory in your scratch or working space
mkdir -p $SLURM_SUBMIT_DIR/data

# Move into the data directory
cd $SLURM_SUBMIT_DIR/data

# Download the file with curl
curl -L -O https://example.com/dataset.zip

# Print confirmation
echo "Download complete: $(ls -lh dataset.zip)"

# we may actually need to use a special transfer node to run this command 
# this may be a Jes question
