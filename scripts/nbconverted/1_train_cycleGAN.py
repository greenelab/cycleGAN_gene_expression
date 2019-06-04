
# coding: utf-8

# # Train cycleGAN
# 
# Scripts to train cycleGAN were adopted from the [cycleGAN repository](https://github.com/eriklindernoren/PyTorch-GAN#cyclegan) that were based on the [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf) paper.
# 
# The scripts were modified to:
# 1. Use fully connected networks as opposed to convoultional neural networks
# 2. Input gene expression data as opposed to image data

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
data_file = f"{os.path.dirname(os.getcwd())}/data/pseudomonas/train_set_normalized.pcl"


# In[3]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0).T
original_shape = data.shape
print(original_shape)
data.head(5)


# ## Train

# In[4]:


get_ipython().run_line_magic('run', 'functions/cyclegan_transcript.py --dataset_name "pseudomonas" --n_epochs 100 --decay_epoch 50 --input_dim 5549 --hidden_dim 1000 --output_dim 100 --num_samples 1191 --batch_size 100')


# ## Plot

# In[5]:


# Read in loss files
G_loss_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data","pseudomonas","train", "G_loss.txt")

D_loss_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data","pseudomonas","train","D_loss.txt")

G_loss_data = pd.read_csv(G_loss_file, header=None, sep=',').T
D_loss_data = pd.read_csv(D_loss_file, header=None, sep=',').T


# In[6]:


G_loss_data.head(5)


# In[7]:


D_loss_data.head(5)


# In[8]:


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


# In[9]:


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

