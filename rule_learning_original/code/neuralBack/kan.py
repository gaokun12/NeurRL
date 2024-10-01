# %%
import sys
import torch
from rule_learning_original.code.pykan.kan import *
import pandas as pd

# %%
model = KAN(width=[3,5,1], grid=5,k=3,seed=0)

# %%
all_data = pd.read_csv('rule_learning_original/code/neuralBack/event_data.csv')
train_data = all_data.sample(frac=0.8, random_state=200)
test_data = all_data.drop(train_data.index)
dataset = {}
dataset['train_label'] = torch.tensor(train_data['label'].values.astype(np.float32)).reshape(-1,1)
dataset['train_input'] = torch.tensor(train_data.drop('label', axis = 1).values.astype(np.float32)) 
dataset['test_label'] = torch.tensor(train_data['label'].values.astype(np.float32)).reshape(-1,1)
dataset['test_input'] = torch.tensor(train_data.drop('label', axis = 1).values.astype(np.float32))
 
dataset['train_input'].shape, dataset['train_label'].shape, dataset['test_input'].shape, dataset['test_label'].shape

# %%
model(dataset['train_input'])
model.plot(beta=100)

# %%
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)

# %%



