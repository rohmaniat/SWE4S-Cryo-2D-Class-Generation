#!/bin/bash
#SBATCH --partition=titan
#SBATCH --job-name=download_data
#SBATCH --output=Data/log_files/download_data_%j.log
#SBATCH --error=Data/log_files/download_data_%j.err
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roto9457@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=32G
##SBATCH --gres=gpu:1

# load any required modules
module purge
module load curl
module list

# Create a data directory in your scratch or working space
mkdir -p $SLURM_SUBMIT_DIR/Data

# Move into the data directory
cd $SLURM_SUBMIT_DIR/Data

# First File

FILE_URL="https://calla.rnet.missouri.edu/cryoppp/11183.tar.gz"
FILE_NAME="11183.tar.gz"

# Download the file with curl
curl -L -O $FILE_URL

# Download all files from the webpage
# wget -r -np -nH --cut-dirs=1 -P data https://calla.rnet.missouri.edu/cryoppp/10061.tar.gz

# Print confirmation
echo "Download complete: $(ls -lh $FILE_NAME)"

# Extract the .tar.gz file
#tar -xvzf "Data/$FILE_NAME"

#echo "Extraction complete. $(ls -lh $FILE_NAME)" 

# Second File

FILE_URL="https://calla.rnet.missouri.edu/cryoppp/11056.tar.gz"
FILE_NAME="11056.tar.gz"

# Download the file with curl
curl -L -O $FILE_URL

# Download all files from the webpage
# wget -r -np -nH --cut-dirs=1 -P data https://calla.rnet.missouri.edu/cryoppp/10061.tar.gz

# Print confirmation
echo "Download complete: $(ls -lh $FILE_NAME)"
i
# Extract the .tar.gz file
#tar -xvzf "Data/$FILE_NAME"

#echo "Extraction complete. $(ls -lh $FILE_NAME)" 

#Finish Output
echo "Process complete"
date
uptime

