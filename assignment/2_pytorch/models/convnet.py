import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        (c, h, w) = im_size
        self.conv1 = nn.Conv2d(c, hidden_dim, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        h_out = int(np.floor((h-kernel_size)+1))
        w_out = int(np.floor((w-kernel_size)+1))
        h_out1 = int(np.floor((h_out-2)/2+1))
        w_out1 = int(np.floor((w_out-2)/2+1))
      
        self.fc = nn.Linear(np.prod((hidden_dim,h_out1, w_out1)), n_classes)        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        
        x = self.pool(F.relu(self.conv1(images)))
        x = x.reshape(x.shape[0],np.prod(x.shape[1:]))
        x = self.fc(x)
        scores = F.softmax(x, dim=1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

