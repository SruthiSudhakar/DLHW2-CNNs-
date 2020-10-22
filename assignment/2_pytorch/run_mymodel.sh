#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 7 \
    --hidden-dim 10 \
    --epochs 4 \
    --weight-decay 0.9 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.01 | tee mymodel.log

#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
