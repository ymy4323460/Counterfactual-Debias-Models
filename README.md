# Introduction

The implementation of causal debiasing methods, including IPS, SNIPS, Direct, Doubly Robust, ATT, CVIB and ACL.

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

For more package details please see environment.yml.
