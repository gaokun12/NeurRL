## Imports
import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
#%matplotlib inline
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd

dataset = MNIST(root = 'MNIST/data/', download = True)
test_dataset = MNIST(root = 'MNIST/data/', train = False)
all_number = [0,1,2,3,4,5,6,7,8,9]
for i in all_number:
    # list remove element
    other_number = all_number.copy()
    other_number.remove(i)
    print(other_number)
    positive_label = [i]
    negative_label = other_number
    first_name = f'p{i}'
    second_name = f'nrest'
    # transfer image to time series data 
    for original in (dataset, test_dataset):
        if len(original) == 60000:
            info = 'train'
        else:
            info = 'test'
        demo_rate = 0.05
        select_instance = int(len(original)*demo_rate)
        ini_instance = 0
        print(f'Working on {len(original)}')
        train_ts = []
        pbar = tqdm(total=len(original))
        for image, label in (original):
            if label in positive_label or label in negative_label:
                image_np = list(np.array(image).reshape(-1))
                if label in positive_label:
                    image_np.append(1)
                elif label in negative_label:
                    image_np.append(0)
                else:
                    raise('Error')
                train_ts.append(image_np)
                ini_instance += 1
            print('select_numbers', ini_instance)
            if ini_instance >= select_instance:
                break
            pbar.update(1)
        print(len(train_ts))
        column = []
        for i in range(28*28):
            column.append('pixel'+str(i))
        column.append('label')
        train_df = pd.DataFrame(train_ts, columns = column)
        print(train_df.head())
        train_df.to_csv(f'MNIST/MNIST_{info}_{first_name}{second_name}_demo.csv', index = False)
