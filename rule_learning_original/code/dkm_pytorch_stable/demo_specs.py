#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import torch
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict


n_clusters = 3 # Number of clusters to obtain


# Pre-process the dataset
# data = data / 255.0 # Normalize the levels of grey between 0 and 1

# Get the split between training/test set and validation set
# test_indices = read_list("split/mnist/test")
# validation_indices = read_list("split/mnist/validation")

# Auto-encoder architecture

hidden_1_size = 500
hidden_2_size = 500
hidden_3_size = 2000
embedding_size = n_clusters




# dimensions = [500,embedding_size,500, # Encoder layer dimensions
#                input_size] # Decoder layer dimensions
# activations = [torch.nn.ReLU(), None, # Encoder layer activations
#                torch.nn.ReLU(),  None] # Decoder layer activations
# names = ['enc_hidden_1', 'embedding', # Encoder layer names
#          'dec_hidden_1', 'output'] # Decoder layer names

