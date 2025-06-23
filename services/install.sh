#!/bin/bash

# Copyright 2024 Queensland University of Technology.
#
# The programming code herein is licensed to The Australian Institute of Marine Science (AIMS)
# by The Queensland University of Technology (QUT) to use for testing and validation of the
# Coral Spawn and Larvae Imaging Camera System (CSLICS).
# 
# All liabilities and guarantees for this program code and its supporting components are as stipulated
# in the relevant agreements relating to CSLICS between AIMS and QUT and by the licenses of the
# supporting components where made by a third party. QUT accepts no liability for modifications made
# to the programming code by parties other than QUT.

# Author: 	Alec Tutin
# Date:		2024-09-26

print_usage() {
    script=$(basename "$0")
	echo "Usage: $script [-b|--broker-host 0.0.0.0] [-i|--image-source ICAM_540|PICAM] [-c|--config-path /path/to/config] [-p|--config-process filename.json] [-p|--config-camera filename.json]"
}

cslics_host="192.168.1.10"
image_source="ICAM_540"
config_path="/home/$USER/cslics_config"
config_process="camera_process_config_icam.json"
config_camera="camera_config_icam_540.json"

while [[ "$#" -gt 0 ]]; do
	case $1 in
		"-b"|"--broker-host" )
			shift; cslics_host="$1"; shift;;
		"-i"|"--image-source" )
			shift; image_source="$1"; shift;;
        "-p"|"--config-path" )
            shift; config_path="$1"; shift;;
        "-t"|"--config-process" )
            shift; config_process="$1"; shift;;
        "-c"|"--config-camera" )
            shift; config_camera="$1"; shift;;
		"-h"|"--help" )
			print_usage; shift;;
		* )
			echo "Invalid option: $1"; print_usage; exit 1;;
	esac
done

project_name="cslics_ref_raas_vision_processor"
package_name="cslics_vision_processor"

include_directories=("$package_name")
include_files=("LICENSE" "client_vision_processor.py")

# Copy project to /opt/cslics
script_directory=`dirname $0`
extract_path=`realpath "$script_directory/.."`
python_interpreter=`which python3`

installation_path="/opt/cslics/${project_name}/"

echo "Copying software to installation directory: $installation_path"

if [ -d "$installation_path" ]; then
	echo "Overwriting existing installation..."
	sudo rm -rf $installation_path
fi

sudo mkdir -p "$installation_path"

for directory in ${include_directories[@]}; do
	sudo cp -R "${extract_path}/${directory}" "$installation_path"
done

for file in ${include_files[@]}; do
	sudo cp "${extract_path}/${file}" "$installation_path"
done

# Copy configuration templates
mkdir -p "$config_path"/models
cp -n "$extract_path"/config/* "$config_path"

# Copy service files to systemd directory
echo "Installing services..."

cd "$script_directory"

export python_interpreter
export cslics_host
export image_source
export config_path
export config_process
export config_camera
export EXIT_STATUS="\$EXIT_STATUS"

cat cslics_camera.service | envsubst | sudo tee /lib/systemd/system/cslics_camera.service > /dev/null

# Enable services
sudo systemctl enable cslics_camera.service --now

echo "Done!"
