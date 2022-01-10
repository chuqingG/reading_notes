# Persistent RNNs

Speed up RNN training to train model on bigger datasets efficiently.

## Motivation

There's a tradeoff on mini-batch size:
- too small: the same recurrent weights need to be loaded for many times
- too big:
  - require more memory when train a network 
  - many GPUs are idle
  - complicates the deployment of models

So try to find a way to **load weights once without increasing the mini-batch size**.

## Method

