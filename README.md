# WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks

![](images/overview.png "Overview of our HPR model.")

## Introduction

This repository includes the PyTorch implementation for our paper 
"WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks".


## Installation

This project is built upon the following environment:
* Python 3.7
* PyTorch 1.7.1
* pytorch_wavelets

The easiest way to install ``pytorch_wavelets`` is to clone the [**repo**](https://github.com/fbcotter/pytorch_wavelets) and pip install
it:

    $ git clone https://github.com/fbcotter/pytorch_wavelets
    $ cd pytorch_wavelets
    $ pip install .

## Train
Train a model on the CIFAR-10 dataset by
```
python train.py --dataset cifar10 
```

## Evaluate
```
python eval.py --dataset cifar10
```

## Acknowledgements

Some of our codes refer to [**Input-Aware Dynamic Backdoor Attack**](https://github.com/VinAIResearch/input-aware-backdoor-attack-release), thanks to their great work!

## Citation

If you find our work insightful or useful, please consider citing:
```
@article{WaveAttack,
title = {WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks},
journal = {Advances in Neural Information Processing Systems},
author = {
    Xia, Jun and Yue, Zhihao and Zhou, Yingbo and Ling, Zhiwei and Shi, Yiyu and Wei, Xian and Chen, Mingsong}
}
```****

