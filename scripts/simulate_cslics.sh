#!/bin/bash

# Public licence for commercial/non-commercial use RRA (internationally)
# 
# Each Project IP Owner grants to, or must obtain for, each other Party and any member of the public a perpetual, irrevocable, worldwide, non-exclusive, royalty-free, non-transferable licence (including a right of sub-license to any person (in the case of GBRF including, but not limited to, the Department)) to Use the Project IP and Project Improvements, in the field of reef restoration and adaptation, for:
# 
#   (a) non-commercial purposes, educational and/or research purposes (including for the performance of Core Commonwealth Functions by the Department); and/or
#   (b) commercial purposes, whether in Australia or elsewhere.
# 
# Each Project IP Owner acknowledges that any licence granted to the Department for Core Commonwealth Functions will not be restricted to use in the field of reef restoration and adaptation.
# 
# Derivative works must be distributed with a copy of this licence which does not further restrict the rights of licensees.
# 
# Amendment providing additional Limitation of Liability for QUT
# 
# QUT does not warrant that:
# 
#   (a) software/code is fit for the Approved Purpose, or that it has any particular qualities or characteristics;
#   (b) the software/code is free from errors, viruses, worms, or similar defects;
#   (c) the use of the software/code by the Licensee will lead to any particular result; or
#   (d) the use of the software/code will not infringe the rights (including Intellectual Property rights) of any person.
# 
# Copyright (C) 2025 Queensland University of Technology
# 

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

