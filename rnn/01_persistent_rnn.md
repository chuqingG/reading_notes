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

## Methods

### Persistent kernel
  - **preemptive multitasking** on GPU, use global synchronization barrier 
  - exit and restart kernel until all kernels succeed
  - try to let more threads than one block run concurently
  
### Alternatives to reduce memory & `batch_size_per_gpu`
  - **model parallelism** 
    - partition recurrent layers among many GPUs
    - No change in `batch size`, but decrease the `batch_size_per_gpu`
    - expensive inter-GPU sync between each `t`, efficient for large layer
  - **truncated back propagation through time**
    - reduce activation memory by performing forward and backward over a fixed timescale
    - **！** lose long time dependency information, reduce accuracy
  - **caching activations in CPU memory** 
    - use CPU off-line memory to cache activation data for backward
    - efficient when data movement is faster than arithmetic operations of forward and backward
    - **！** reduce the bandwidth of all-reduce, which slowdown the data-parallelism across nodes
