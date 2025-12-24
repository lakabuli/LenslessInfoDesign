#!/bin/bash
# Script to download the RML model from Google Drive
# Downloads the model from reconstruction/experiments_rml in the shared folder

# Google Drive folder ID
FOLDER_ID="129L_sWBN9wNy7yLfXQds_5uPFE36QFYO"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing gdown..."
    pip install gdown
fi

# Create a temporary directory for downloading
TEMP_DIR=$(mktemp -d)
echo "Downloading from Google Drive to temporary directory..."

# Download the entire folder from Google Drive
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}?usp=share_link" --output "${TEMP_DIR}"

# Find and copy the experiments_rml folder contents to current directory
if [ -d "${TEMP_DIR}/reconstruction/experiments_rml" ]; then
    echo "Copying model files to current directory..."
    cp -r "${TEMP_DIR}/reconstruction/experiments_rml"/* .
    echo "Download complete! Model files are in the current directory."
else
    echo "Warning: Could not find reconstruction/experiments_rml in downloaded folder."
    echo "Please check the Google Drive folder structure."
    echo "Downloaded files are in: ${TEMP_DIR}"
fi

# Clean up temporary directory
rm -rf "${TEMP_DIR}"

