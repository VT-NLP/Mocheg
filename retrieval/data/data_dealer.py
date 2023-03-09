

import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import pandas as pd 
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import torch
from PIL import Image 
from util.read_example import get_relevant_document_dir, load_corpus_df, read_image
from transformers import CLIPTokenizer



# from util.read_example import get_relevant_document_dir
def get_relevant_document_dir(splited_dir):
    from pathlib import Path
    path = Path(splited_dir)
    whole_dataset_dir=path.parent.absolute()
    return whole_dataset_dir
  
    

class DataDealer:
    def __init__(self,mode=None) :
        self.mode=mode
    def load_corpus(self,data_folder, corpus_max_size):
        pass 
class ImageDataDealer:
    def __init__(self)  :
       
        # self.model._first_module().max_seq_length =77
        self.tokenizer_for_truncation=  CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        
    def load_corpus(self,data_folder, corpus_max_size):
        corpus={}
        news_dict={}
        relevant_document_dir=get_relevant_document_dir(data_folder)
      
        news_dict,relevant_document_img_list=read_image(relevant_document_dir,news_dict,content="img")
        image_corpus=os.path.join(relevant_document_dir,"images")
        for relevant_document_img in relevant_document_img_list:
            corpus[relevant_document_img]=os.path.join(image_corpus,relevant_document_img)
        return corpus
    
    def load_queries(self,data_folder,needed_qids): 
        dev_queries_file = os.path.join(data_folder, 'Corpus2_for_retrieval.csv')
        df_news = pd.read_csv(dev_queries_file ,encoding="utf8")
        
        dev_queries = {}        #Our dev queries. qid => query
        for _,row in tqdm.tqdm(df_news.iterrows()):
            claim_id=row["claim_id"]
            query=row['Claim']
            if claim_id in needed_qids:
                query=self.truncate_text(query)
                dev_queries[claim_id]=query.strip() 
        ### Load data
    
        return dev_queries
    
    def truncate_text(self,text):
        tokens=self.tokenizer_for_truncation([text])
        decoded_text=self.tokenizer_for_truncation.decode(tokens.input_ids[0][:75],skip_special_tokens =True)
        # decoded_text=decoded_text.replace("<|startoftext|>","")
        # decoded_text=decoded_text.replace("<|endoftext|>","")
        # sequence = self.tokenizer_for_truncation.encode_plus(text, add_special_tokens=False,   
        #                                        max_length=77, 
        #                                        truncation=True, 
        #                                        return_tensors='pt' )
        # return self.tokenizer_for_truncation.decode(sequence.input_ids.detach().cpu().numpy().tolist()[0])
        return decoded_text 
             
             

def gen_corpus_embedding(corpus,model):
    batch_size=480 #480
    live_num_in_current_batch=0
    live_num=0
    current_image_batch=[]
    total_img_emb= torch.tensor([],device= torch.device('cuda'))
    corpus_len=len(corpus)
    for corpus_id,corpus_img_path in corpus.items():
        image=Image.open(corpus_img_path)
        current_image_batch.append(image)
        live_num_in_current_batch+=1
        
        if live_num_in_current_batch%batch_size==0:
            img_emb = model.encode(current_image_batch, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
            total_img_emb=torch.cat([total_img_emb,img_emb],0)
            live_num_in_current_batch=0
            current_image_batch=[]
            live_num+=batch_size
            print(live_num/corpus_len)
    return total_img_emb
def gen_corpus_embedding_with_cache(corpus,model,data_folder,use_precomputed_corpus_embeddings,media):
    corpus_folder= get_relevant_document_dir(data_folder)
    emb_folder=os.path.join(corpus_folder,f"embed")
    emb_filename = 'img_corpus_emb.pkl'
    emb_dir=os.path.join(emb_folder,emb_filename)
    if use_precomputed_corpus_embeddings and os.path.exists(emb_dir): 
        with open(emb_dir, 'rb') as fIn:
            emb_file = pickle.load(fIn)  
            img_emb, img_names =emb_file["img_emb"],emb_file["img_names"] 
        print("Images:", len( img_names))
    else:
        img_emb=gen_corpus_embedding(corpus,model)
        emb_file = { "img_emb":  img_emb, "img_names": list(corpus.keys()) }            #,"img_folder":img_folder
        pickle.dump( emb_file, open(emb_dir , "wb" ) )
    return img_emb
             
class TextDataDealer(DataDealer):
     
    
    def load_corpus(self,data_folder, corpus_max_size):
        corpus = {}             #Our corpus pid => passage
       
        # df_news = pd.read_csv(collection_filepath ,encoding="utf8")
        corpus_df=load_corpus_df(data_folder)
        # Read passages
        for _,row in tqdm.tqdm(corpus_df.iterrows()):
            pid=str(row["claim_id"])+"-"+str(row["relevant_document_id"])+"-"+str(row["paragraph_id"])
            passage=row["paragraph"]
            if   corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
                corpus[pid] = passage.strip()
        return corpus
          
    def load_queries(self,data_folder,needed_qids):
        
        dev_queries_file = os.path.join(data_folder, 'Corpus2.csv')
        df_news = pd.read_csv(dev_queries_file ,encoding="utf8")
        
        dev_queries = {}        #Our dev queries. qid => query
        for _,row in tqdm.tqdm(df_news.iterrows()):
            claim_id=row["claim_id"]
            claim=row['Claim']
            if claim_id in needed_qids:
                dev_queries[claim_id]=claim.strip() 
        ### Load data
    
        return dev_queries

# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.obtain_corpus_by_idx(pos_id) 
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text =self.obtain_corpus_by_idx(neg_id)   
        query['neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def obtain_corpus_by_idx(self,corpus_id):
        return self.corpus[corpus_id]

    def __len__(self):
        return len(self.queries)    
    

class MSMARCOImageDataset(MSMARCODataset):
    def __init__(self, queries, corpus):
        super().__init__(queries,corpus)
        
    def obtain_corpus_by_idx(self,corpus_id):
        
        image=Image.open(self.corpus[corpus_id])
        return image
    
    




def read_queries(data_folder):
    ### Read the train queries, store in queries dict
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query
    return queries
 


def get_train_queries(queries,positive_rel_docs,negative_rel_docs):
    train_queries = {}
    for qid,pid_set in positive_rel_docs.items():
        
        train_queries[qid] = {'qid': qid, 'query': queries[qid], 'pos': pid_set, 'neg': negative_rel_docs[qid]}
    return train_queries    



def get_content(corpus,corpus_id_list):
    corpus_content_list=[]
    for corpus_id in corpus_id_list:
        corpus_content_list.append(corpus[corpus_id])
    return corpus_content_list

def get_queries_with_content(queries,positive_rel_docs,negative_rel_docs,corpus):
    train_queries = {}
    for qid,pid_set in positive_rel_docs.items():
        negative_corpus_id_list=negative_rel_docs[qid]
        
        train_queries[qid] = {'qid': qid, 'query': queries[qid], 'positive': get_content(corpus,pid_set)
                              , 'negative': get_content(corpus,negative_corpus_id_list)}
    return train_queries  


 
def load_cross_encoder_samples(data_folder,queries,corpus,num_max_negatives ):    
     
    qrel_file_name="qrels.csv"
   
    qrels_filepath = os.path.join(data_folder, qrel_file_name)
    df_news = pd.read_csv(qrels_filepath ,encoding="utf8")
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need
    negative_rel_docs={}
    dev_rel_docs = {}       #Mapping qid => set with relevant pids
    train_samples=[]
    negative_num_dict={}
    negative_train_samples=[]
    positive_train_samples=[]
    
    # Load which passages are relevant for which queries
    for _,row in tqdm.tqdm(df_news.iterrows()):
        
        qid,  pid, relevance= row["TOPIC"],row["DOCUMENT#"],row["RELEVANCY"]
        query = queries[qid]
        input_example=InputExample(texts=[query, corpus[pid]], label=relevance)
        if relevance==0:
            if qid in negative_num_dict:
                cur_num=negative_num_dict[qid]
                if cur_num>num_max_negatives:
                    continue
                else:
                    negative_num_dict[qid]+=1
            else:
                negative_num_dict[qid]=1
            negative_train_samples.append(input_example)
        else:
            positive_train_samples.append(input_example)
        train_samples.append(input_example)
    print(f"{len(positive_train_samples), len(negative_train_samples)}")
    train_samples,pos_weight=balance(positive_train_samples,negative_train_samples,train_samples)
         
    return train_samples ,pos_weight

def balance(positive_train_samples,negative_train_samples,train_samples):
    ratio= int(len(negative_train_samples)/len(positive_train_samples))
    
    if ratio>1:
        for i in range(ratio-1):
            train_samples.extend(positive_train_samples)
    
        new_float_ratio=len(negative_train_samples)/(len(positive_train_samples)*ratio)    
    else:
        new_float_ratio=len(negative_train_samples)/len(positive_train_samples)
    return train_samples,new_float_ratio
    