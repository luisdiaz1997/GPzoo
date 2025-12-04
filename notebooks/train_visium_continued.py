import torch
from torch import optim
from gpzoo.kernels import MGGP_NSF_RBF
from gpzoo.gp import MGGP_NSF
import squidpy as sq
import numpy as np
from gpzoo.utilities import train



model_name = 'visium_nsf'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

adata = sq.datasets.visium_hne_adata()

Y_sums = np.array(np.sum(adata.raw.X > 0, axis=0))[0]
Y = np.array(adata.raw.X[:, Y_sums>200].todense(), dtype=int).T
X = adata.obsm['spatial']

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)


L =10
M = 120
groupsX = torch.tensor(adata.obs.cluster.values.codes).type(torch.LongTensor)
n_groups = len(adata.obs.cluster.values.categories)

kernel = MGGP_NSF_RBF(sigma=10.0, lengthscale=10.0, group_diff_param=20.0, n_groups=n_groups, device=device)

model = MGGP_NSF(X=X, y=Y, kernel=kernel, M=M, L=L, jitter=1e-2, n_groups=n_groups)
model.load_state_dict(torch.load(model_name, map_location=device))
model.to(device)


X_train = X.to(device)
Y_train = Y.to(device)
groupsX = groupsX.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

losses = train(model, optimizer, X_train, groupsX, Y_train, device, steps=10000)


torch.save(model.state_dict(), model_name)