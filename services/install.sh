#!/bin/bash

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

# Copy project to /opt/cslics
script_directory=`dirname $0`
project_path=`realpath "$script_directory/.."`
project_name=`basename "$project_path"`
python_interpreter=`which python3`

echo "Copying software to installation directory: /opt/cslics/$project_name"
sudo mkdir -p /opt/cslics/
sudo cp -R "$project_path" /opt/cslics/

# Copy configuration templates
mkdir -p ~/cslics_config/models
cp -n "$project_path"/config/* ~/cslics_config/

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
