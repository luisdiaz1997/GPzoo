import torch

def _squared_dist(X, Z):
    
    X2 = (X**2).sum(1, keepdim=True)
    Z2 = (Z**2).sum(1, keepdim=True)
    XZ = X.matmul(Z.t())
    r2 = X2 - 2 * XZ + Z2.t()
    return r2.clamp(min=0)

def add_jitter(K, jitter=1e-3):
    if K.dim()==2:
        N, _ =  K.shape
        K.view(-1)[::N+1] += jitter
        return K

    if K.dim()==3:
        L, N, _ = K.shape
        K.view(L, -1)[:, ::N+1] += jitter
        
        return K
    

def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


def _embed_distance_matrix(distance_matrix):

    # Code adapted from https://github.com/andrewcharlesjones/multi-group-GP
    N = len(distance_matrix)
    D2 = distance_matrix**2
    C = torch.eye(N) - 1/N *torch.ones(size=(N, N))
    B = -0.5*C @ D2 @ C
    L, Q = torch.linalg.eigh(B)
    embedding = Q @ torch.diag(_torch_sqrt(L, 1e-6))
    return embedding