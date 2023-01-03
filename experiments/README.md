# Experiments

The files in this directory can be used to reproduce the results in the [original paper](https://arxiv.org/abs/2301.00557). The contents of this directory include:

- `baselines/` - code for implementing several baseline methods
- `tabular/` - code for reproducing experiments with three tabular datasets (spam, miniboone, diabetes)
- `mnist/` - code for MNIST experiments
- `cifar/` - code for CIFAR-10 experiments

You'll need to install a couple additional packages to run all the baseline methods:

- `sage` - can be downloaded from [here](https://github.com/iancovert/sage)
- `captum` - see installation instructions [here](https://captum.ai/#quickstart)
- `tqdm`

The three experimental directories (`tabular`, `mnist` and `cifar`) each contain a shell script called `run.sh` that can run all the methods for each dataset. This will take a long time to finish, so it's recommended to start by running one method at a time. For example, you can run GDFS on the spam dataset as follows:

```bash
cd tabular/
python run.py --dataset=spam --method=greedy
```
