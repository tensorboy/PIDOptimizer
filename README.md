# PIDOptimizer (Proportional–Integral–Derivative Optimizer)
This repository contains source code of the paper:
* [*A PID Controller Approach for Stochastic Optimization of Deep Networks*](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf) (CVPR 2018)

## Prerequisite: 
* matplotlib==2.0.2

## Train MLP on MNIST DATAST
`python mnist_pid.py`
`python mnist_momentum.py`
`python compare.py`

<div align="center">
  <img src="moment_vs_pid.jpg" width="700px" />
  <p>PID Vs. SGD-Momentum</p>
</div>

## Citation:
If PIDOptimizer is used in your paper/experiments, please cite the following paper.
```
@inproceedings{pid2018,
   title={A PID Controller Approach for Stochastic Optimization of Deep Networks},
   author={Wangpeng An and Haoqian Wang and Qingyun Sun and Jun Xu and Qionghai Dai and Lei Zhang},
   booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   month = {June},
   year={2018}
}
```
