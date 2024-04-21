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

# engine = create_engine(f'postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/nppes')
conn = psycopg2.connect(host='localhost', user='postgres',
                              password='mysecretpassword', 
                              dbname='nppes', port=5432)
# connection = engine.connect()
# Register the vector type with psycopg2
register_vector(conn)

cur=conn.cursor()
cur.execute('select npi,provider_specialty from public.provider_data_nppes')
# print(cur.fetchall())
specialties=[(res[0],res[-1].split('|')) for i,res in enumerate(cur.fetchall())]
npis,specialties=zip(*specialties)
specialties=[[x[0],x[2]] for x in specialties]
print(specialties)

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

bs=128
all_embs = []
for i in tqdm(np.arange(0, len(specialties), bs)):
    # toks = tokenizer.batch_encode_plus(specialties[i:i+bs], 
    #                                    padding="max_length", 
    #                                    max_length=25, 
    #                                    truncation=True,
    #                                    return_tensors="pt")
    toks = tokenizer.__call__(specialties[i:i+bs], 
                                       padding="max_length", 
                                       max_length=25, 
                                       truncation=True,
                                       return_tensors="pt",
                                       is_split_into_words =True)

    toks_cuda = {}
    for k,v in toks.items():
        toks_cuda[k] = v
    cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
    all_embs.append(cls_rep.cpu().detach().numpy())

all_embs = np.concatenate(all_embs, axis=0)
all_embs=[(np.array(x)) for x in all_embs]
for npi, embedding in zip(npis, all_embs):
    cur.execute('INSERT INTO embeddings (npi, embedding) VALUES (%s, %s)', (npi, embedding))

conn.commit()
# print(all_embs)
# print("INSERT INTO embeddings (embedding) VALUES (%s)", (all_embs[0]))
# cur.execute("INSERT INTO embeddings (embedding) VALUES (%s)", (all_embs[0]))
# execute_values(cur, "INSERT INTO embeddings (embedding) VALUES %s", all_embs[0])


# for res in results.fetchmany(10):
#     row=[res._asdict()['provider_specialty']]
#     bs=128
#     all_embs = []
#     for i in tqdm(np.arange(0, len(row), bs)):
#         toks = tokenizer.batch_encode_plus(row[i:i+bs], 
#                                            padding="max_length", 
#                                            max_length=25, 
#                                            truncation=True,
#                                            return_tensors="pt")
#         toks_cuda = {}
#         for k,v in toks.items():
#             toks_cuda[k] = v
#         cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
#         all_embs.append(cls_rep.cpu().detach().numpy())

#     all_embs = np.concatenate(all_embs, axis=0)
#     print(all_embs.shape)


# # replace with your own list of entity names
# all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 

# bs = 128 # batch size during inference
# all_embs = []
# for i in tqdm(np.arange(0, len(all_names), bs)):
#     toks = tokenizer.batch_encode_plus(all_names[i:i+bs], 
#                                        padding="max_length", 
#                                        max_length=25, 
#                                        truncation=True,
#                                        return_tensors="pt")
#     toks_cuda = {}
#     for k,v in toks.items():
#         toks_cuda[k] = v
#     cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
#     all_embs.append(cls_rep.cpu().detach().numpy())

# all_embs = np.concatenate(all_embs, axis=0)
# print(all_embs.shape)