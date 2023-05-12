#!/bin/bash

experiment_name=cpg+nes

for body in "babya" "babyb" "blokky" "garrix" "gecko" "insect" "linkin" "longleg" "penguin" "pentapod" "queen" "salamander" "squarish" "snake" "spider" "stingray" "tinlicker" "turtle" "ww" "zappa"

morphologies = ['T6', 'T10', 'T14','gecko6', 'gecko10', 'gecko14','snake6', 'snake10', 'snake14','spider6', 'spider10','spider14']
do
	for num in {1..10}
	do
		screen -d -m -S "${experiment_name}" -L -Logfile "./${experiment_name}.log" nice -n19 python3 "optimize.py" $body $num &
#		python DRL/PPO/optimize.py $body $num &
	done
done
