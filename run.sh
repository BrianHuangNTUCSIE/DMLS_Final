#!/bin/bash
# IID
bash IID_cifar10.sh LR &
bash IID_cifar10.sh CNN &
wait
bash IID_cifar10.sh DNN
bash IID_cifar10.sh DenseNet

bash IID_cifar100.sh LR &
bash IID_cifar100.sh CNN &
wait
bash IID_cifar100.sh DNN
bash IID_cifar100.sh DenseNet

bash IID_mnist.sh LR &
bash IID_mnist.sh CNN &
wait
bash IID_mnist.sh DNN
bash IID_mnist.sh DenseNet

# Non-IID
bash non_IID_cifar10.sh LR &
bash non_IID_cifar10.sh CNN &
wait
bash non_IID_cifar10.sh DNN
bash non_IID_cifar10.sh DenseNet

bash non_IID_cifar100.sh LR &
bash non_IID_cifar100.sh CNN &
wait
bash non_IID_cifar100.sh DNN
bash non_IID_cifar100.sh DenseNet

bash non_IID_mnist.sh LR &
bash non_IID_mnist.sh CNN &
wait
bash non_IID_mnist.sh DNN
bash non_IID_mnist.sh DenseNet