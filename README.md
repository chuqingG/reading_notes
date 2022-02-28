# Reading notes

> This repo is not up to date.

### Deep Learning Compiler

- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](compiler/01_tvm.md), [*paper link*](https://www.usenix.org/conference/osdi18/presentation/chen)

- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](compiler/02_ansor.md), [*paper link*](https://arxiv.org/abs/2006.06762)

### CPU optimizations

- Optimizing CNN Model Inference on CPUs, [*paper link*](https://www.usenix.org/system/files/atc19-liu-yizhi.pdf)

### GPU optimizations

- [Matmul Optimization](gpu/01_matmul.md)
  
### RNN optimizations

- [Persistent RNNs](./rnn/01_persistent_rnn.md), [*blog link*](http://svail.github.io/persistent_rnns/)
- [Optimizing RNN performance](./rnn/03_optimizing_rnn_performance.md), [*blog link*](http://svail.github.io/rnn_perf/)
- Optimizing RNNs with Differentiable Graphs, [*blog link*](http://svail.github.io/diff_graphs/)
-  ~~Echo: Compiler-based GPU Memory Footprint Reduction for LSTM RNN Training~~, [*paper link*](http://www.cs.toronto.edu/~pekhimenko/Papers/Echo-ISCA_20.pdf)

- Optimizing Recurrent Neural Networks in cuDNN 5, [*blog link*](https://developer.nvidia.com/blog/optimizing-recurrent-neural-networks-cudnn-5/)
<!-- - [Optimizing Recurrent Neural Networks in cuDNN 5](./rnn/04_optimizing_rnn_in_cudnn5.md), [*blog link*](https://developer.nvidia.com/blog/optimizing-recurrent-neural-networks-cudnn-5/) -->

### Parallel & Vectorizing

- [A Loop Transformation Theory and an Algorithm to Maximize Parallelism](./parallel/02_loop_transformation.md), [*paper link*](https://suif.stanford.edu/papers/wolf91b.pdf)
- [Auto-vectorizing TensorFlow Graphs: Jacobians, Auto-batching and Beyond](./parallel/01_auto_vectorizing.md), [*paper link*](https://arxiv.org/pdf/1903.04243.pdf)

### Decentralized algorithms


### Autonomous Driving

- Too Afraid to Drive: Systematic Discovery of Semantic DoS Vulnerability in Autonomous Driving Planning under Physical-World Attacks, [*paper link*](https://arxiv.org/pdf/2201.04610.pdf)
- Invisible for both Camera and LiDAR: Security of Multi-Sensor Fusion based Perception in Autonomous Driving Under Physical-World Attacks, [*paper link*](http://me.ningfei.org/paper/msf-adv.pdf)