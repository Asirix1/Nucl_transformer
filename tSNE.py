from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import h5py

dset1=[]
labels=[]
for i in ['intergenic_regions','exons','introns','5UTR','3UTR']:
 with h5py.File('mean_embedings_1_'+i+'.hdf5', 'r') as f:
    dset=f['mean_embedings_1']
    dset1+=dset
    labels+=100*[i]
dset1 = pd.DataFrame(dset1)
f.close()

embed = TSNE(n_components=2, perplexity=10, early_exaggeration=12, learning_rate=200, n_iter=5000, n_iter_without_progress=300, min_grad_norm=0.0000001, init='random', metric='euclidean')

X_embedded = embed.fit_transform(dset1)
tsne_results=pd.DataFrame(X_embedded, columns=['tsne1', 'tsne2'])
print(tsne_results)
plt.scatter(tsne_results[0:99]['tsne1'], tsne_results[0:99]['tsne2'], c='red')
plt.scatter(tsne_results[100:199]['tsne1'], tsne_results[100:199]['tsne2'], c='blue')
plt.scatter(tsne_results[200:299]['tsne1'], tsne_results[200:299]['tsne2'], c='green')
plt.scatter(tsne_results[300:399]['tsne1'], tsne_results[300:399]['tsne2'], c='orange')
plt.scatter(tsne_results[400:499]['tsne1'], tsne_results[400:499]['tsne2'], c='purple') 
plt.legend(['intergenic_regions','exons','introns','5UTR','3UTR'])
plt.savefig('tSNE.png')

