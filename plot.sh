#!/bin/bash
python3 plot_IID.py cifar10 LR
python3 plot_IID.py cifar10 CNN
python3 plot_IID.py cifar10 DNN
python3 plot_IID.py cifar10 DenseNet

python3 plot_IID.py cifar100 LR
python3 plot_IID.py cifar100 CNN
python3 plot_IID.py cifar100 DNN
python3 plot_IID.py cifar100 DenseNet

python3 plot_IID.py mnist LR
python3 plot_IID.py mnist CNN
python3 plot_IID.py mnist DNN
python3 plot_IID.py mnist DenseNet

python3 plot_non_IID.py cifar10 LR
python3 plot_non_IID.py cifar10 CNN
python3 plot_non_IID.py cifar10 DNN
python3 plot_non_IID.py cifar10 DenseNet

python3 plot_non_IID.py cifar100 LR
python3 plot_non_IID.py cifar100 CNN
python3 plot_non_IID.py cifar100 DNN
python3 plot_non_IID.py cifar100 DenseNet

python3 plot_non_IID.py mnist LR
python3 plot_non_IID.py mnist CNN
python3 plot_non_IID.py mnist DNN
python3 plot_non_IID.py mnist DenseNet

python3 merge_plot.py