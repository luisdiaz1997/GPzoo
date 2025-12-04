#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import matplotlib.pyplot as plt
from torch import optim, distributions, nn
from gpzoo.kernels import MGGP_NSF_RBF
from gpzoo.gp import MGGP_NSF
import squidpy as sq
import numpy as np
from gpzoo.utilities import  train_batched


# In[2]:

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device




adata = sq.datasets.slideseqv2()
Y_sums = np.array(np.sum(adata.raw.X > 0, axis=0))[0]
Y = np.array(adata.raw.X[:, Y_sums>1000].todense() * 1000, dtype=int).T


# In[5]:


X = adata.obsm['spatial']




# In[7]:


X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)



# In[10]:


L =10
M = 300
groupsX = torch.tensor(adata.obs.cluster.values.codes).type(torch.LongTensor)
n_groups = len(adata.obs.cluster.values.categories)

kernel = MGGP_NSF_RBF(sigma=2.0, lengthscale=30.0, group_diff_param=4.0, n_groups=n_groups, device=device)

model = MGGP_NSF(X=X, y=Y, kernel=kernel, M=M, L=L, jitter=1e-2, n_groups=n_groups)

# idz = torch.multinomial(torch.ones(X.shape[0]), num_samples=M*n_groups, replacement=False)
# model.svgp.Z = nn.Parameter(X[idz])
# model.svgp.groupsZ = nn.Parameter(groupsX[idz].type(torch.LongTensor), requires_grad=False)


# In[11]:


model.load_state_dict(torch.load('slideseq_mggp_test.pth'))

model.to(device)


# In[12]:


#freeze kernel parameters first, unfreeze for finetunning
model.svgp.kernel.lengthscale.requires_grad = False
model.svgp.kernel.group_diff_param.requires_grad = False
model.svgp.kernel.sigma.requires_grad = False
model.svgp.Z.requires_grad=False


# In[13]:


X_train = X.to(device)
Y_train = Y.to(device)
groupsX = groupsX.to(device)


# In[14]:


optimizer = optim.Adam(model.parameters(), lr=1e-3)


# In[15]:


losses = train_batched(model, optimizer, X_train, groupsX, Y_train, device, steps=40*60*8, batch_size=1000, E=10)


# In[16]:


plt.plot(losses)


# In[17]:


torch.save(model.state_dict(), 'slideseq_mggp_test.pth')
