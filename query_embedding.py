import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel  
import psycopg2 as psycopg2
from sqlalchemy import create_engine,text
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy
from psycopg2.extensions import register_adapter, AsIs
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.float32, addapt_numpy_float32)
register_adapter(numpy.int64, addapt_numpy_int64)

def get_engine():
    postgres_con = f'postgresql+psycopg2://{MRF_STATS_USER}:{MRF_STATS_PASS}@{MRF_STATS_HOST}/{MRF_STATS_DB}'
    return create_engine(postgres_con)


conn = psycopg2.connect(host='localhost', user='postgres',
                              password='mysecretpassword', 
                              dbname='nppes', port=5432)

register_vector(conn)

cur=conn.cursor()
cur.execute('select npi,provider_specialty from public.provider_data_nppes')


tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
query="Heart Doctor"
toks = tokenizer.__call__(query, 
                                   padding="max_length", 
                                   max_length=25, 
                                   truncation=True,
                                   return_tensors="pt",
                                   )

print(toks)
all_embs = []
toks_cuda = {}
for k,v in toks.items():
    toks_cuda[k] = v
cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
all_embs.append(cls_rep.cpu().detach().numpy())
cur.execute("""SELECT pdn.* from provider_data_nppes pdn inner join (select * from embeddings order by embedding <->  %s LIMIT 5) e using (npi)""", (all_embs[0][0],))


print(cur.fetchall())
