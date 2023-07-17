import os
import scipy as sc
import nucleotide_transformer
import h5py

import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model

model_name = '2B5_1000G'

parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name=model_name,
    mixed_precision=False,
    embeddings_layers_to_save=(1,),
    attention_maps_to_save=((1, 4), (7, 18)),
    max_positions=1000   
)
forward_fn = hk.transform(forward_fn)

sequences=[]
with open("/beegfs/data/hpcws/ws1/popov-transformer_work/genomic_reg_fa/exons.bed.fa") as f:
 for line in f: 
    if not line.startswith(">") and len(line)<=1000:
        sequences+=[line.rstrip()]
    if len(sequences)>=100:
        print(len(sequences))
        break
f.close()

tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

random_key = jax.random.PRNGKey(0)

outs = forward_fn.apply(parameters, random_key, tokens)

embeddings = outs["embeddings_1"][:, 1:, :]  # removing CLS token
padding_mask = jnp.expand_dims(tokens[:, 1:] != tokenizer.pad_token_id, axis=-1)
masked_embeddings = embeddings * padding_mask  # multiply by 0 pad tokens embeddings
sequences_lengths = jnp.sum(padding_mask, axis=1)
mean_embeddings = jnp.sum(masked_embeddings, axis=1) / sequences_lengths
print(mean_embeddings.shape)

with h5py.File('mean_embedings_1.hdf5', 'w') as f:
    dset = f.create_dataset("mean_embedings_1", data=mean_embeddings)
f.close()

