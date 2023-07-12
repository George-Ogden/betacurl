# BetaCurl
Attempts at acheiving superhuman curling in a simulated curling environment (https://github.com/George-Ogden/curling). 
## Installation
```
pip install -r requirements.txt
```
You also need to install mujoco bindings from https://github.com/openai/mujoco-py
## Running
The scripts use [WandB](https://wandb.ai/). If you do not want this feature enabled, `export WANDB_MODE=disabled`. 
To run the curling environment:
```
python train_curling.py --warm_start_games 2000 --num_games_per_episode 10 --num_iterations 1000 --save_directory curling --save_frequency 10 --num_simulations 200 --training_epochs 5 --training_patience 0 --project_name curling
```
To run the mujoco environment:
```
python train_mujoco.py --domain_name cartpole --task_name swingup --save_directory cartpole-swingup --save_frequency 5 --max_rollout_depth 0 --num_simulations 50 
```
To preview the curling model:
```
python preview_curling.py --model_directory curling/model-last
```
To preview the mujoco model:
```
python preview_mujoco.py --model_directory cartpole-swingup/model-best/ --domain_name cartpole --task_name swingup
```
