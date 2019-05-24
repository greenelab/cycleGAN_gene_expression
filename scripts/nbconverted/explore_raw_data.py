
# coding: utf-8

# # About the data

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import pandas as pd
import numpy as np
import random
#import umap
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

metadata_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "metadata","sample_annotations.tsv")


# In[3]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip').T
original_shape = data.shape
print(original_shape)
data.head(5)


# In[5]:


# Read in metadata
metadata = pd.read_table(metadata_file, header=0, sep='\t', index_col=0)
metadata_shape = metadata.shape
print(metadata_shape)
metadata.head(5)


# ## What does the clustering look like between planktonic vs biofilm in Gene space
# 
# How much heterogeneity exists between samples?
# How many genes contribute to the difference?

# In[ ]:


# Label samples by planktonic vs biofilm

# Since our simulation set all genes in set A to be the same value for a give sample
# we can consider a single gene in set A to query by
rep_gene_A = geneSetA.iloc[0][0]
geneA_exp = sim_data[rep_gene_A]

sample_id = sim_data.index

# Bin gene A expression
geneA_exp_labeled = sim_data.assign(
    rep_geneA=(
        list( 
            map(
                lambda x:
                '1' if 0 < x and x <=0.1 
                else '2' if 0.1< x and x <=0.2 
                else '3' if 0.2<x and x<=0.3
                else '4' if 0.3<x  and x<=0.4
                else '5' if 0.4<x and x<=0.5
                else '6' if 0.5<x and x<=0.6
                else '7' if 0.6<x and x<=0.7
                else '8' if 0.7<x and x<=0.8
                else '9' if 0.8<x and x<=0.9
                else '10',
                geneA_exp
            )      
        )
    )
)
geneA_exp_labeled = geneA_exp_labeled.astype({"rep_geneA": int})
geneA_exp_labeled.head()


# Plot gene expression in gene space
# Each dot is a sample. Each sample is colored based on its expression of gene A
# 
# In the legend 1 ~ gene A expression is (0.0, 0.1], 2 ~ gene A expression is (0.1, 0.2], etc.

# In[ ]:


# UMAP embedding of raw gene space data
embedding = umap.UMAP().fit_transform(sim_data)
embedding.shape


# In[ ]:


# UMAP plot of raw gene expression data
geneA_exp_labeled = geneA_exp_labeled.assign(sample_index=list(range(geneA_exp_labeled.shape[0])))
for x in geneA_exp_labeled.rep_geneA.sort_values().unique():
    plt.scatter(
        embedding[geneA_exp_labeled.query("rep_geneA == @x").sample_index.values, 0], 
        embedding[geneA_exp_labeled.query("rep_geneA == @x").sample_index.values, 1], 
        c=sns.color_palette()[x-1],
        alpha=0.7,
        label=str(x)
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of gene expression data in GENE space', fontsize=14)
plt.legend()
plt.savefig(geneSpace_file, dpi=300)

