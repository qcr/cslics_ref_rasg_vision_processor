#!/bin/bash

# Author: 	Alec Tutin
# Date:		2024-09-26

# Disable services
sudo systemctl disable cslics_camera.service --now

# Remove service files from systemd directory
sudo rm /lib/systemd/system/cslics_camera.service

# Delete project from /opt/cslics
script_directory=`dirname $0`
project_path=`realpath "$script_directory/.."`
project_name=`basename "$project_path"`
install_path="/opt/cslics/$project_name"

echo "Removing software from installation directory: $install_path"
sudo rm -R "$install_path"

echo "Done!"
