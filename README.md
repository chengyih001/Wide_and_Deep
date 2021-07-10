# Simple Wide and Deep Model
Author: Yih Cheng

This repository is a simple implementation of [Heng-Tze Cheng, *et al.* Wide & Deep Learning for Recommender Systems (2016)](https://arxiv.org/abs/1606.07792).<br>
The dataset used is from [UCI Machine Learning Repository: Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult).

## Environment
> Python == 3.9.5
>
> keras == 2.4.3
>
> keras-preprocessing == 1.1.2

## Overview
As recommender systems become more and more popular within social media and platforms, respective researches have also grown rapidly. However, these systems are intrinsicly different from traditional deep learning models due to the unique construction of input data. With sparse data contributing a large part of the input, normal methods of designing and constructing models has lost its effectiveness, and methods such as embedding lookup have become popular when dealing with similar scenarios. In addition, both memorization and generalization are required in recommender systems, as memorization provides predictions based on user's past behavior while generalization determines to expand the aspects of predictions. In paper [Heng-Tze Cheng, *et al.* Wide & Deep Learning for Recommender Systems (2016)](https://arxiv.org/abs/1606.07792), wide components for memorization and deep components for generalization have been concatenated together for better recommendations. The Wide and Deep model proposed has proven its value in industries, being widely implemented and utilized. It has provided actual boosts in revenues for many applications and platforms.

## Dataset
There are two datasets in "./dataset", respectively `adult.train` and `adult.test`. Both includes the same categorical and continuous columns.

    CATEGORICAL_COLUMNS = [
        'workclass', 'education', 'marital_status', 'occupation', 
        'relationship', 'race', 'sex', 'native_country'
    ]

    CONTINUOUS_COLUMNS = [
        'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'
    ]

There is also a column for `income_bracket`, stating whether the person's income is >50K or not. In this repository, the goal is to try and use categorical and continuous inputs to predict whether a person's income is larger than 50K or not.

## Modes
* Wide mode : Linear model with cross product between categorical items as input
* Deep mode : Standard deep learning models with embedded categorical items and continuous items concatenated as input
* Wide and Deep mode : A hybrid between wide component and deep component by concatenating the respective output layers together

## Validation Results
* Wide mode : loss = 0.5024 | accuracy = 0.7479
* Deep mode : loss = 0.5099 | accuracy = 0.7970
* Wide and Deep mode : loss = 0.4339 | accuracy = 0.8199
