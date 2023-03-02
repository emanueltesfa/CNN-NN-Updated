from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3, stride=1, padding=0, name="conv1"),
            MaxPoolingLayer(2,1, name="maxpool1"),
            flatten(name="flatten1"),
            fc(27, 5, init_scale=0.02, name="fc1"),
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            #stide 1 
            ConvLayer2D(input_channels=3, kernel_size=5, number_filters=30, stride=2, padding=0, name="conv"),
            gelu(name="gelu2"),
            MaxPoolingLayer(3,1, name="maxpool"),
            flatten(name="flatten"),
            fc(1080, 500, init_scale=0.02, name="fc"),
            gelu(name='gelu3'),
            dropout(0.5,seed=seed, name="dropout"),
            fc(500, 250, init_scale=0.02, name="fc2"),
            gelu(name="last_gelu"),
            dropout(0.6, seed=seed, name="dropout2"),
            fc(250, 20, init_scale=0.02, name="fc3"),
            gelu(name="gelu")
            
            
            ########### END ###########
        )