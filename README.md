# Greedy dynamic feature selection (GDFS)

GDFS is a method for dynamically selecting features based on conditional mutual information. It was developed in the paper [Learning to Maximize Mutual Information for Dynamic Feature Selection](https://arxiv.org/abs/2301.00557).

In the dynamic feature selection (DFS) problem, you handle examples at test-time as follows: you begin with no features, progressively select a specific number of features (according to a pre-specified budget), and then make predictions given the available information. The problem can be addressed in many different ways, but GDFS tries to greedily select features according to their conditional mutual information (CMI) with the response variable. CMI is difficult to calculate, so GDFS approximates these selections using a custom training approach.

## Installation

You can get started by cloning the repository, and then pip installing the package in your Python environment as follows:

```bash
pip install .
```

## Usage

GDFS involves learning two separate networks: one responsible for making predictions (the predictor) and one responsible for making selections (the policy). Both networks receive a subset of features as their input, and the policy outputs probabilities for selecting each feature. During training, we sample a random feature using the Concrete distribution, but at test time we simply use the argmax.

The diagram below illustrates the training approach:

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/iancovert/dynamic-selection/main/docs/concept.jpg"/>
</p>

For usage examples, see the following:

- [Spam](https://github.com/iancovert/dynamic-selection/blob/main/notebooks/spam.ipynb): a notebook showing an example with the UCI spam detection dataset.
- [MNIST](https://github.com/iancovert/dynamic-selection/blob/main/notebooks/mnist.ipynb): a notebook example with MNIST (digit recognition).
- [MNIST-Grouped](https://github.com/iancovert/dynamic-selection/blob/main/notebooks/mnist-grouped.ipynb): shows how to use feature grouping, which is necessary for some datasets (e.g., when using one-hot encoded categorical features).
- The [experiments](https://github.com/iancovert/dynamic-selection/tree/main/experiments) directory contains code to reproduce experiments from the original paper

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Wei Qiu
- Mingyu Lu
- Nayoon Kim
- Nathan White
- Su-In Lee

## References

Ian Covert, Wei Qiu, Mingyu Lu, Nayoon Kim, Nathan White, Su-In Lee. "Learning to Maximize Mutual Information for Dynamic Feature Selection." *ICML, 2023*.
