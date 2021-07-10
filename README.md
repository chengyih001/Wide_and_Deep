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

## Modes
* Wide : Linear model with cross product between categorical items as input
* Deep : Standard deep learning models with embedded categorical items and continuous items concatenated as input
* Wide and Deep : A hybrid between wide component and deep component by concatenating the respective output layers together
