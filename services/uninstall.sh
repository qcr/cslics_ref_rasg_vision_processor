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

# Disable services
sudo systemctl disable cslics_camera.service --now

# Remove service files from systemd directory
sudo rm /lib/systemd/system/cslics_camera.service

# Delete project from /opt/cslics
script_directory=`dirname $0`
project_name="cslics_ref_raas_vision_processor"
install_path="/opt/cslics/$project_name"

echo "Removing software from installation directory: $install_path"
sudo rm -R "$install_path"

echo "Done!"
