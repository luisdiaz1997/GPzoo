import squidpy as sq
import numpy as np
from gpzoo.utilities import regularized_nmf, dims_autocorr
import matplotlib.pyplot as plt
import sys 



def main(random_seed):
    adata = sq.datasets.slideseqv2()
    Y_int = np.array(adata.raw.X[:].todense()*100, dtype=int).T
    X = adata.obsm['spatial']
    L = 20
    F, W = regularized_nmf(Y_int.T, L=L, shrinkage=0.3, max_iter = 1000, solver='mu', init='nndsvdar', beta_loss='kullback-leibler', random_state=random_seed)
    moran_idx, moranI = dims_autocorr(np.exp(F), X)
    loadings = np.exp(F.T)[moran_idx]
    max_val = np.percentile(loadings, 99)
    min_val = np.percentile(loadings, 1)


    size = 7
    plt.figure(figsize=(size*5, size*4), tight_layout=True)
    for i in range(L):
        plt.subplot(L//5, 5, i+1)
        plt.scatter(X[:, 0], X[:,1], c=loadings[i], vmin=min_val, vmax=max_val, alpha=0.9, s=5, cmap='turbo')
            
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_facecolor('xkcd:gray')

    plt.savefig('../figures/slideseq_nmf/slideseq_v2_nmf_seed_'+str(random_seed)+'.png', dpi=200)


if __name__ == "__main__":
    random_seed = int(sys.argv[1])
    main(random_seed)