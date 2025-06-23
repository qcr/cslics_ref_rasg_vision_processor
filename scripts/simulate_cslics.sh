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

# Author: Alec Tutin
# Date: 2024-08-28

session_name="simulate_cslics"

tmux new-session -s $session_name -d

tmux split-window -t $session_name -v

tmux send -t $session_name:0.0 "python3 -m cslics_vision_processor.testing.simulate_cameras $*" ENTER

tmux send -t $session_name:0.1 "./$(dirname "$0")/simulate_lights.sh" ENTER

installations_running=0

while true; do
	read -p "$installations_running simulated installations running. Press Enter to start another and CTRL+D to quit!"
	
	if [[ $? -ne 0 ]]; then
		break
	fi
	
	tmux send -t $session_name:0.0 ENTER
	tmux send -t $session_name:0.1 ENTER
	
	installations_running=$((installations_running + 1))
done

echo; echo "Shutting down..."

tmux send -t $session_name:0.0 C-D
tmux send -t $session_name:0.1 C-D

tmux kill-session -t $session_name

