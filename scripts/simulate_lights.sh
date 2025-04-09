#!/bin/bash

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

