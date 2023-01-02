# Set variables
GPU=7
NUM_TRIALS=5

python run_greedy.py --gpu=$GPU --num_trials=$NUM_TRIALS
python run_random.py --gpu=$GPU --num_trials=$NUM_TRIALS
python run_center.py --gpu=$GPU --num_trials=$NUM_TRIALS
