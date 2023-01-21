# Set variables
GPU=4
NUM_RESTARTS=3
NUM_TRIALS=5

for METHOD in sage permutation deeplift intgrad cae greedy
do
    python run.py --method=$METHOD --gpu=$GPU --num_trials=$NUM_TRIALS --num_restarts=$NUM_RESTARTS
done
