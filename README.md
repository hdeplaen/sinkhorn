# Sinkhorn for PyTorch

This repository contains an implementation of Sinkhorn's algorithm in LibTorch, for faster runs when regularly called. As for now, four variants of Sinkhorn's algorithm have been implemented:
- `base`: Canonical Sinkhorn without any tweaks.
- `unbalanced`: Unbalanced version controlled by `tau1` and `tau2`.
- `stable`: A more stable implementation in log-space. The computation is more heavy than the `stable` version.
- `unblalanced_stable`: Combination of `unbalanced` and `stable`.

## Benchmark
![CPU benchmark](./data/cpu1.png)
![GPU benchmark](./data/gpu1.png)

*Each point is the result of 150 experiments on different histograms and cost matrices, each time performed 10 times (so 1500 runs per datapoint in total). The standard deviation is given in the shaded area. The number of histogram varies (x-axis) and their size is fixed at 50 and 30. The number of iterations is also fixed at 128.* 

## Build
To build the file for your system, insure first that you have the C++ frontend libraries installed. Then run `python setup.py build_ext`, which will create a cpython file corresponding to your OS, architecture and Python version in `build/lib`. It suffices to move that file wherever you want and import it as a classical python file. For it to work, always ensure that `import torch` is called before `import sinkhorn`.

## TO DO
(if I have some time over)
- [x] Build various checks to verify its good working
- [x] Benchmark it against other implementations both on CPU and GPU.
- [ ] Build a documentation
- [ ] Deploy it on PyPi
