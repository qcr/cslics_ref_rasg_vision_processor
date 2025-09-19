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

generate_id() {
	RANDOM=$1
	id=""

	for i in {0..3}; do
		segment=$RANDOM # It is important to not call $RANDOM within `` as the value will no longer be seeded...
		id+=`printf "%04X" $segment`
	done
	
	echo $id
}

session_name="simulate_cslics_lights"

RANDOM=1337
declare -a light_pids

if [ -z $CSLICS_HOST ]; then
	CSLICS_HOST="localhost"
fi

while true; do
	num_lights=${#light_pids[@]}
	read -p "$num_lights simulated lights running. Press Enter to start another and CTRL+D to quit!"
	
	if [[ $? -ne 0 ]]; then
		break
	fi
	
	seed=$RANDOM
	light_id=`generate_id $seed`
	
	echo "Starting light with ID: $light_id connected to $CSLICS_HOST..."
	
	mosquitto_pub -h $CSLICS_HOST -t cslics/lights/connected -m "$light_id" --repeat 99999 --repeat-delay 10 &
	
	light_pids[$num_lights]=$!
done

echo; echo "Shutting down..."
	
for pid in ${light_pids[@]}; do
	kill $pid
done

