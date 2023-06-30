

def add_jitter(K, jitter=1e-3):
    if K.dim()==2:
        N, _ =  K.shape
        K.view(-1)[::N+1] += jitter
        return K

    if K.dim()==3:
        L, N, _ = K.shape
        K.view(L, -1)[:, ::N+1] += jitter
        
        return K