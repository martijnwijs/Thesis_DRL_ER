# Thesis Deep Reinforcement Learning on Evolutionary Robotics github repository 

PPO learning algorithm on 12 different morphologies on the task of gait learning and rotation 
for robot locomotion using Mujoco (preferred) and Isaacgym based Revolve2 simulator.
the code is mainly a duplication from the following repository  https://github.com/onerachel/Controllers_Learners

## Installation 
simple steps to install are:
``` 

1. git clone https://github.com/onerachel/revolve2
2. cd revolve2/
   git clone https://github.com/martijnwijs/Thesis_DRL_ER
3. cd ..
4. virtualenv -p python3.8 .venv
5. source .venv/bin/activate
6. ./dev_requirements.sh
``` 
## running
1. cd revolve2
2. cd Thesis_DRL_ER
3. cd Controllers_learners
4. python run_Gecko.py (or 1 of the run files of the other morphologies)
## Documentation 

[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/) 
