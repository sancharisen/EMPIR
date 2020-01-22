# CleverHans_EMPIR for evaluating robustness of ensembles of mixed precision DNN models

<img src="https://github.com/tensorflow/cleverhans/blob/master/assets/logo.png?raw=true" alt="cleverhans logo">

[![Build Status](https://travis-ci.org/tensorflow/cleverhans.svg?branch=master)](https://travis-ci.org/tensorflow/cleverhans)

This repository contains the source code for the paper EMPIR: Ensembles of Mixed Precision Deep Networks for Increased Robustness against Adversarial Attacks ([Accepted at ICLR 2020](https://openreview.net/forum?id=HJem3yHKwH))

It is based on CleverHans, a Python library to
benchmark machine learning systems' vulnerability to
[adversarial examples](http://karpathy.github.io/2015/03/30/breaking-convnets/).
You can learn more about such vulnerabilities on the accompanying [blog](http://cleverhans.io).

## Setting up
+ Install Tensorflow 
+ Git clone this repository
+ Set environment variable PYTHONPATH to path of this repository

## Results
<table>
    <tr align="center">
        <th rowspan="2">Dataset</th>
        <th rowspan="2">Ensemble Type</th>
        <th colspan=3>Precisions</th>
        <th rowspan=2>Unperturbed Accuracy (%)</th>
        <th colspan=4>Adversarial Accuracy (%)</th>
    </tr>
    <tr align="center">
        <th>Model 1</th>
        <th>Model 2</th>
        <th>Model 3</th>
        <th>CW</th>
        <th>FGSM</th>
        <th>BIM</th>
        <th>PGD</th>
    </tr>
    <tr align="center">
       <td>MNIST</td>
       <td> </td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a> </td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a> </td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a> </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
    </tr>
    <tr align="center">
       <td>CIFAR-10</td>
       <td> </td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a></td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a> </td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a> </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
    </tr>
    <tr align="center">
       <td>ImageNet</td>
       <td> </td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a></td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a> </td>
       <td> abits=4, wbits=2 <a href="https://github.com/sancharisen/cleverhans_EMPIR">Download</a> </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
       <td> 100 </td>
    </tr>
</table>

## Example commands
+ `python cleverhans_tutorials/mnist_attack.py --nb_samples=10000 --attack_iterations=50 --wbits=$model1_weight_prec --abits=$model1_activation_prec --wbits2=$model2_weight_prec --abits2=$model2_activation_prec --ensembleThree --model_path1=/path/to/model1/ckpt --model_path2=/path/to/model2/ckpt --model_path3=/path/to/model3/ckpt` - White-Box CW attack on MNIST EMPIR model
+ `python cleverhans_tutorials/mnist_attack.py --nb_samples=10000 --attack_iterations=50 --model_path=/path/to/baseline/model/ckpt` - White-Box CW attack on MNIST baseline model

### Dependencies

This library uses [TensorFlow](https://www.tensorflow.org/) to accelerate graph
computations performed by many machine learning models.
Installing TensorFlow is therefore a pre-requisite.

You can find instructions
[here](https://www.tensorflow.org/install/).
For better performance, it is also recommended to install TensorFlow
with GPU support (detailed instructions on how to do this are available
in the TensorFlow installation documentation).

Installing TensorFlow will
take care of all other dependencies like `numpy` and `scipy`.

### Installation

Once dependencies have been taken care of, you can install CleverHans using
`pip` or by cloning this Github repository.

#### `pip` installation

If you are installing CleverHans using `pip`, run the following command:

```
pip install -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans
```

#### Manual installation

If you are installing CleverHans manually, you need to install TensorFlow
first. Then, run the following command to clone the CleverHans repository
into a folder of your choice:

```
git clone https://github.com/tensorflow/cleverhans
```

On UNIX machines, it is recommended to add your clone of this repository to the
`PYTHONPATH` variable so as to be able to import `cleverhans` from any folder.

```
export PYTHONPATH="/path/to/cleverhans":$PYTHONPATH
```

You may want to make that change permanent through your shell's profile.

### Currently supported setups

Although CleverHans is likely to work on many other machine configurations, we
currently [test it](https://travis-ci.org/tensorflow/cleverhans) with Python
{2.7, 3.5} and TensorFlow {1.0, 1.1} on Ubuntu 14.04.5 LTS (Trusty Tahr).

## Citing this work

If you use CleverHans for academic research, you are highly encouraged
(though not required) to cite the following [paper](https://arxiv.org/abs/1610.00768):

```
@article{papernot2016cleverhans,
  title={cleverhans v1.0.0: an adversarial machine learning library},
  author={Papernot, Nicolas and Goodfellow, Ian and Sheatsley, Ryan and Feinman, Reuben and McDaniel, Patrick},
  journal={arXiv preprint arXiv:1610.00768},
  year={2016}
}
```
There is not yet an ArXiv tech report for v2.0.0 but one will be prepared soon.

## Copyright

Copyright 2017 - Google Inc., OpenAI and Pennsylvania State University.
