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
import sys 

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.float32, addapt_numpy_float32)
register_adapter(numpy.int64, addapt_numpy_int64)

conn = psycopg2.connect(host='localhost', user='postgres',
                              password='mysecretpassword', 
                              dbname='nppes', port=5432)

cur=conn.cursor()
register_vector(conn)

def get_query_embedding(query):
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    toks = tokenizer.__call__(query, 
                                       padding="max_length", 
                                       max_length=4, 
                                       truncation=True,
                                       return_tensors="pt",
                                       )

    all_embs = []
    toks_cuda = {}
    for k,v in toks.items():
        toks_cuda[k] = v
    cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
    all_embs.append(cls_rep.cpu().detach().numpy())
    return all_embs[0][0]

def get_embedding_matches(query):
    query_embedding=get_query_embedding(query)
    cur.execute("""SELECT pnn.*,pdn.provider_specialty from provider_data_nppes pdn inner join (select * from embeddings order by embedding <->  %s LIMIT 5) e using (npi) inner join provider_names_nppes pnn on (pdn.npi=pnn.npi)""", (query_embedding,))
    results=cur.fetchall()
    return results

def get_specialty_matches_using_embeddings(query):
    query_embedding=get_query_embedding(query)
    cur.execute("""SELECT concat(split_part(pdn.provider_specialty,'|',1),' ',split_part(pdn.provider_specialty,'|',3)) from provider_data_nppes pdn inner join (select * from embeddings order by embedding <->  %s LIMIT 5) e using (npi) inner join provider_names_nppes pnn on (pdn.npi=pnn.npi) limit 2""", (query_embedding,))
    results=cur.fetchall()
    results = [x[0] for x in results]
    return results


if __name__=='__main__':
    results=get_specialty_matches_using_embeddings(sys.argv[1])
    print(results)
