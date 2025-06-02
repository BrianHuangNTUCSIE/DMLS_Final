#!/bin/bash
bash non_IID_cifar10.sh LR &
bash non_IID_cifar10.sh CNN &
wait
bash non_IID_cifar10.sh DNN
bash non_IID_cifar10.sh DenseNet