
# coding: utf-8

# # Train cycleGAN
# 
# Scripts to train cycleGAN were adopted from [https://github.com/eriklindernoren/PyTorch-GAN#cyclegan] and modified to 1) use fully connected networks as opposed to convoultional neural networks and 2) input gene expression data as opposed to image data

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Load arguments
data_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data","pseudomonas","train", "A", "all-pseudomonas-gene-normalized.zip")


# In[3]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip').T
original_shape = data.shape
print(original_shape)
data.head(5)


# ## Train

# In[4]:


get_ipython().run_line_magic('run', 'cyclegan_transcript.py --dataset_name "pseudomonas" --n_epochs 100 --decay_epoch 50 --input_dim 5549 --hidden_dim 1000 --output_dim 100 --num_samples 1191 --batch_size 100')


# ## Plot

# In[6]:


# Read in loss files
G_loss_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data","pseudomonas","train", "G_loss.txt")

D_loss_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data","pseudomonas","train","D_loss.txt")

G_loss_data = pd.read_csv(G_loss_file, header=None, sep=',').T
D_loss_data = pd.read_csv(D_loss_file, header=None, sep=',').T


# In[7]:


G_loss_data


# In[8]:


D_loss_data


# In[9]:


# Generator loss
G_loss_out_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data","pseudomonas","train", "G_loss_plot.jpg")

fig = plt.figure()
plt.plot(G_loss_data.index, G_loss_data.values)
fig.suptitle('Generator Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Generator Loss')
fig.savefig(G_loss_out_file)


# In[10]:


# Discriminator loss
D_loss_out_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data","pseudomonas","train","D_loss_plot.jpg")

fig = plt.figure()
plt.plot(D_loss_data.index, D_loss_data.values)
fig.suptitle('Discriminator Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Discriminator Loss')
fig.savefig(D_loss_out_file)

