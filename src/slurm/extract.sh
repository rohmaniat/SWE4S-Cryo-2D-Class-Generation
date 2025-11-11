#!/bin/bash
#SBATCH --partition=titan
#SBATCH --job-name=extract_tar_gz
#SBATCH --output=Data/log_files/extract_tar_gz_%j.out
#SBATCH --error=Data/log_files/extract_tar_gz_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --time=03:00:00

# Define the path to your .tar.gz file
ARCHIVE_FILE_1="Data/11051.tar.gz"
ARCHIVE_FILE_2="Data/11056.tar.gz"
ARCHIVE_FILE_3="Data/11057.tar.gz"
ARCHIVE_FILE_4="Data/11183.tar.gz"

# Extract the .tar.gz file
tar -xvzf "$ARCHIVE_FILE_1" -C "Data/"

echo "Extraction complete."

# SECOND FILE

# Extract the .tar.gz file
tar -xvzf "$ARCHIVE_FILE_2" -C "Data/"

echo "Extraction complete."

# THIRD FILE

# Extract the .tar.gz file
tar -xvzf "$ARCHIVE_FILE_3" -C "Data/"

echo "Extraction complete."

# FOURTH FILE

# Extract the .tar.gz file
tar -xvzf "$ARCHIVE_FILE_4" -C "Data/"

echo "Extraction complete."
~                              
