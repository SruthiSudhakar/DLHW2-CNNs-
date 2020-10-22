import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        (c, h, w) = im_size
        kernel_size_2 = 5
        kernel_size_3 = 3
        pool = 2
        self.conv1 = nn.Conv2d(c, hidden_dim, kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size_2)
        self.conv3 = nn.Conv2d(20, 30, kernel_size_3)
        self.pool = nn.MaxPool2d(pool, pool)
        h_out = int(np.floor((h-kernel_size)+1))
        w_out = int(np.floor((w-kernel_size)+1))
        h_out1 = int(np.floor((h_out-pool)/pool+1))
        w_out1 = int(np.floor((w_out-pool)/pool+1))
        
        h_out2 = int(np.floor((h_out1-kernel_size_2)+1))
        w_out2 = int(np.floor((w_out1-kernel_size_2)+1))
        h_out3 = int(np.floor((h_out2-pool)/pool+1))
        w_out3 = int(np.floor((w_out2-pool)/pool+1))

        h_out4 = int(np.floor((h_out3-kernel_size_3)+1))
        w_out4 = int(np.floor((w_out3-kernel_size_3)+1))
        h_out5 = int(np.floor((h_out4-pool)/pool+1))
        w_out5 = int(np.floor((w_out4-pool)/pool+1))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(np.prod((30,h_out5, w_out5)), n_classes)
        self.model_ft = models.resnet152(pretrained=True)   
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, n_classes)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
        # TODO: Implement the forward pass.
        #############################################################################
        '''x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.reshape(x.shape[0],np.prod(x.shape[1:]))
        x = self.fc(x)'''
        x = self.model_ft(images)
        scores = F.softmax(x, dim=1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

