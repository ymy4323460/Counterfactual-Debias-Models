# Introduction

The implementation of causal debiasing methods on recommendation system, including IPS, SNIPS, Direct, Doubly Robust, ATT, CVIB and ACL. 

Contributor: [Mengyue Yang](https://scholar.google.com/citations?user=kJJkqdcAAAAJ&hl=en)

# Methods Provide

[Direct Method (Direct)](https://aclanthology.org/D17-1272/): The counterfactual data is filled by a learned model before model optimization. In this model, the label of counterfactual data is estimated by a model learned from observational data.

[Inverse Propensity Score (IPS)](https://proceedings.mlr.press/v37/swaminathan15.html): The samples are re-weighted to convert the observed data distribution to the ideal unbiased one. The method is achieved by calculating propensity score over observational data first.

[Self-normalized IPS (SNIPS)](https://proceedings.neurips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html): An extension of IPS, where the learning objective variance is reduced by a normalization strategy.

[Doubly Robust (DR)](https://arxiv.org/abs/1103.4601): A combination of IPS and Direct Methods.

[ATT](https://arxiv.org/pdf/1910.01444.pdf): An extension of the direct method based on meta-learning, which leverages two imputation models to double-check the correctness of counterfactual labels.

[CVIB](https://proceedings.neurips.cc/paper/2020/hash/13f3cf8c531952d72e5847c4183e6910-Abstract.html): A debiased method based on information bottleneck.

[ACL](https://arxiv.org/abs/2012.02295): A debiased recommender model based on adversarial training.

# Requirements

python=3.9.7

pytorch=1.10.0=py3.9_cuda11.3_cudnn8.2.0_0

numpy=1.21.2

pandas==1.3.5

scikit-learn=1.0.1

For more package details please see environment.yml.

# Quick start

### Generate synthetic data.

Using the code below can help you generate synthetic data containing 50000 samples and 32 context dimensions.

`python data_simulator_nonlinear_inv.py`

### Run models

We provide the script code to run all the methods.

`sh train_synthetic_baselines.sh`

The script includes the pre-train step to learn the imputation model for Direct-based methods and learn the propensity score evaluator for IPM-based methods.

In the training step, for each model, we show the result of training on the non-uniform data (biased data) and testing on the uniform data (unbias data). The metrics include AUC, ACC, NDCG, Logloss, Recall and Precision.

# Detail settings

### Synthetic data generator

The generation policy is summarized in the following steps, details of the description are shown in section 5.5.1 of the paper [CBR](https://dl.acm.org/doi/pdf/10.1145/3580594). 
- Generate user and item features from a uniform distribution.
- For each user, we generate a non-uniform impression list by user and item features and generate a uniform impression list from a random policy.
- Calculate the user feedback for each item in the impression list.

The generated data directory contains train and dev data, each including the data sample file in CSV format. The user and item features are recorded in files "user_features.npy" and "item_features.npy". 


In the basic synthetic data generation code "<span style="background-color: yellow;">data_simulator_nonlinear_inv.py</span>" you can specify the sample size and the feature dimension. We also provide the implementation of linear and non-linear function to generate user feedbacks.

To change the sample size and feature dimension, please change the related number in the code from "data_simulator_nonlinear_inv.py"

<pre>
```python
# sample size
for sam in [50000]:
    # feature dimension
    for cdim in [32]:
        logit_impression_list_new(mode='dev', policy='nonlinearinv', sample_num=sam, context_dim=cdim)
        random_impression_list(mode='dev', policy='nonlinearinv', sample_num=sam, context_dim=cdim)
```
</pre>



In the advanced generation code "<span style="background-color: yellow;">data_simulator_with_confounder_bias.py</span>", you can specify two additional parameters:

- "bs" represents the degree of bias, where a larger value indicates a lower level of bias in the impression list.
- "gm" represents the degree of confounding interference, where a smaller value indicates a lower level of interference from confounders.

<pre>
```python
# sample size
for sam in [50000]:
    # feature dimension
    for cdim in [32]:
        for gm in [0.5]:
            for bs in [0.5]:
                logit_impression_list_new(mode='dev', step=sam, policy='nonliearlogit', sample_num=sam,
                                          context_dim=cdim, gamma=gm, bs=bs)

                random_impression_list(mode='dev', step=sam, policy='nonliearlogit', sample_num=sam, context_dim=cdim,
                                       gamma=gm, bs=bs)
```
</pre>
