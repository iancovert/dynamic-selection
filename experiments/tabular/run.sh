# Set variables
GPU=0
NUM_RESTARTS=3
NUM_TRIALS=5

# Spam dataset
for METHOD in sage permutation deeplift intgrad cae iterative eddi greedy
do
    python run.py --dataset=spam --method=$METHOD --gpu=$GPU --num_trials=$NUM_TRIALS --num_restarts=$NUM_RESTARTS
done

# Diabetes dataset
for METHOD in sage permutation deeplift intgrad cae iterative eddi greedy
do
    python run.py --dataset=diabetes --method=$METHOD --gpu=$GPU --num_trials=$NUM_TRIALS --num_restarts=$NUM_RESTARTS
done

# Miniboone dataset
for METHOD in sage permutation deeplift intgrad cae iterative eddi greedy
do
    python run.py --dataset=miniboone --method=$METHOD --gpu=$GPU --num_trials=$NUM_TRIALS --num_restarts=$NUM_RESTARTS
done
