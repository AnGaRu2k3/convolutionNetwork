#!/bin/bash

pip install torch==2.3.0 torchvision==0.18.0 matplotlib numpy PyYAML torchsummary gdown
# Example runs with different configurations
# Run 1: ANN model on MNIST and cnn_fashionmnist.pt
python main.py --train --device cuda --model netConfigs/3base.yaml --model-type ann --dataset MNIST --epochs 50 --batch-size 64 --base-lr 0.005 --target-lr 0.00001 --warmup-epochs 5 --save-path models/ann_mnist.pt
python main.py --train --device cuda --model netConfigs/3base.yaml --model-type ann --dataset FashionMNIST --epochs 50 --batch-size 64 --base-lr 0.005 --target-lr 0.00001 --warmup-epochs 5 --save-path models/cnn_fashionmnist.pt
# caltech101
