#!/bin/bash

# Author: 	Alec Tutin
# Date:		2024-09-26

CSLICS_HOST="192.168.1.10"

# Copy project to /opt/cslics
script_directory=`dirname $0`
project_path=`realpath "$script_directory/.."`
project_name=`basename "$project_path"`

echo "Copying software to installation directory: /opt/cslics/$project_name"
sudo mkdir -p /opt/cslics/
sudo cp -R "$project_path" /opt/cslics/

# Copy configuration templates
mkdir -p ~/cslics_config/models
cp -n "$project_path"/config/* ~/cslics_config/

# Copy service files to systemd directory
echo "Installing services..."

cd "$script_directory"
export CSLICS_HOST
cat cslics_camera.service | envsubst | sudo tee /lib/systemd/system/cslics_camera.service > /dev/null

# Enable services
sudo systemctl enable cslics_camera.service --now

echo "Done!"
