from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import h5py
import os

number_sequences=os.environ['number_sequences'] 
layer=os.environ['layer'] 

dset1=[]
labels=[]
for i in ['intergenic_regions','exons','introns','5UTR','3UTR']:
 with h5py.File('mean_embedings_'+str(layer)+'_'+i+'.hdf5', 'r') as f:
    dset=f['mean_embedings_'+str(layer)]
    dset1+=dset
dset1 = pd.DataFrame(dset1)
f.close()
embed = TSNE(n_components=2, perplexity=10, early_exaggeration=12, learning_rate=250, method='barnes_hut', random_state=1, n_iter=5000, n_iter_without_progress=300, min_grad_norm=0.0000001, angle=0.1, init='random', metric='euclidean', n_jobs=-1)

X_embedded = embed.fit_transform(dset1)
tsne_results=pd.DataFrame(X_embedded, columns=['tsne1', 'tsne2'])
i=int(number_sequences)
start=0
count=0
for colors in ['red', 'blue', 'green', 'orange', 'purple']:  
 end=i+i*count-1
 plt.scatter(tsne_results[start:end]['tsne1'], tsne_results[start:end]['tsne2'], c=colors)
 start+=i
 count+=1
plt.legend(['intergenic_regions','exons','introns','5UTR','3UTR'])
plt.savefig('tSNE_layer_'+str(layer)+'_number_sequences_'+str(i)+'.png')
plt.clf() 
