"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.
Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""
from PIL import Image
import pandas as pd 
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import torch
 
from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import logging
import sys
import os
import tarfile
from retrieval.eval.evaluator import MultiMediaInformationRetrievalEvaluator
from retrieval.training.model import MultiMediaSentenceTransformer
from retrieval.utils.enums import TrainAttribute
from util.common_util import setup_with_args
from util.read_example import   load_corpus_df
from retrieval.data.data_dealer import ImageDataDealer, TextDataDealer, gen_corpus_embedding 
from util.read_example import get_relevant_document_dir
 
def load_corpus(data_folder, corpus_max_size):
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
    
def load_qrels(data_folder,media="txt"):    
    if media=="txt":
        qrel_file_name="text_evidence_qrels_sentence_level.csv"
    else:
        qrel_file_name="img_evidence_qrels.csv"#img_evidence_relevant_document_mapping.csv
    qrels_filepath = os.path.join(data_folder, qrel_file_name)
    df_news = pd.read_csv(qrels_filepath ,encoding="utf8")
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need
    negative_rel_docs={}
    dev_rel_docs = {}       #Mapping qid => set with relevant pids
    # Load which passages are relevant for which queries
    for _,row in tqdm.tqdm(df_news.iterrows()):
        
        qid,  pid, relevance= row["TOPIC"],row["DOCUMENT#"],row["RELEVANCY"]

        if relevance==1:

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()
            dev_rel_docs[qid].add(pid)

            needed_pids.add(pid)
            needed_qids.add(qid)
        else:
            if qid not in negative_rel_docs:
                negative_rel_docs[qid] = set()
            negative_rel_docs[qid].add(pid)
    return dev_rel_docs,needed_pids,needed_qids,negative_rel_docs
            
def load_queries(data_folder,needed_qids):
    
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
 
def test_retrieval(model_name,data_folder,corpus,model_save_path,media,use_precomputed_corpus_embeddings): 
    ####  Load model
    model = MultiMediaSentenceTransformer(model_name)

    ir_evaluator=gen_evaluator(data_folder,corpus,media,model,use_precomputed_corpus_embeddings,model_save_path)
    
    ir_evaluator(model, output_path=model_save_path )
    



def gen_corpus_embedding_with_cache(corpus,model,data_folder,use_precomputed_corpus_embeddings,media):
    corpus_folder= get_relevant_document_dir(data_folder)
    emb_folder=os.path.join(corpus_folder,f"supplementary")
    emb_filename = 'img_corpus_emb.pkl'
    emb_dir=os.path.join(emb_folder,emb_filename)
    corpus_dict_path=os.path.join(emb_folder,"corpus_dict.pkl")
    if use_precomputed_corpus_embeddings and os.path.exists(emb_dir): 
        with open(emb_dir, 'rb') as fIn:
            emb_file = pickle.load(fIn)  
            img_emb, img_names =emb_file["img_emb"],emb_file["img_names"] 
        print("Images:", len( img_names))
        if os.path.exists(corpus_dict_path):
            with open(corpus_dict_path, 'rb') as fIn:
                corpus = pickle.load(fIn)  
            print("cached corpus_dict:", len( corpus))
        else:
            print(f"Error! must have corpus_dict.pkl in {emb_folder} while use img_corpus_emb.pkl in {emb_folder}")
            exit()
    else:
        img_emb=gen_corpus_embedding(corpus,model)
        emb_file = { "img_emb":  img_emb, "img_names": list(corpus.keys()) }            #,"img_folder":img_folder
        pickle.dump( emb_file, open(emb_dir , "wb" ) )
        pickle.dump( corpus, open(corpus_dict_path , "wb" ) )
        
    
    return img_emb,corpus


def gen_evaluator(data_folder,corpus,media,model,use_precomputed_corpus_embeddings,model_save_path):
    dev_rel_docs,needed_pids,needed_qids,_=load_qrels(data_folder, media)
    if  media=="txt":
        data_dealer=TextDataDealer()
    else:
        data_dealer=ImageDataDealer()
    dev_queries=data_dealer.load_queries(data_folder,needed_qids)
    
    
    ## Run evaluator
    logging.info("Queries: {}".format(len(dev_queries)))
    logging.info("Corpus: {}".format(len(corpus)))
    save_query_result_path=os.path.join(model_save_path,f"query_result_{media}.pkl")
    if media=="img":
        corpus_embedding,corpus=gen_corpus_embedding_with_cache(corpus,model,data_folder,use_precomputed_corpus_embeddings,media)
        ir_evaluator =MultiMediaInformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[5,10,100,1000  ],
                                                            name="msmarco dev",
                                                            corpus_embed=corpus_embedding,
                                                            score_functions ={'cos_sim': util.cos_sim },
                                                            save_query_result_path=save_query_result_path,
                                                            map_at_k=[5,10],
                                                            ndcg_at_k=[5,10])
    else:
        ir_evaluator = MultiMediaInformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[5,10,100,1000  ],
                                                            name="msmarco dev",
                                                            score_functions ={'cos_sim': util.cos_sim },
                                                            save_query_result_path=save_query_result_path,
                                                            map_at_k=[5,10],
                                                            ndcg_at_k=[5,10])
    
    return ir_evaluator

def get_args():
    
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_max_size", default=0, type=int)
    parser.add_argument("--model_name",default='multi-qa-MiniLM-L6-cos-v1') 
    parser.add_argument("--data_folder",default='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test') #retrieval/input/msmarco-data
    args = parser.parse_args()

    print(args)
    return args


def test(args):
    train_attribute=TrainAttribute[args.train_config]
    if args.model_name is None:
        args.model_name =train_attribute.model_name
    if args.train_batch_size is None:
        args.train_batch_size = train_attribute.batch_size
    if args.epochs is None:
        args.epochs  =train_attribute.epoch
    if args.media is None:
        args.media=train_attribute.media 
    if args.max_seq_length is None:
        args.max_seq_length=train_attribute.max_seq_length
    if  args.media=="txt":
        data_dealer=TextDataDealer()
    else:
        data_dealer=ImageDataDealer()
    corpus_max_size=0
    corpus=data_dealer.load_corpus(args.test_data_folder,  corpus_max_size)
    model_save_path,args=setup_with_args(args,'retrieval/output/runs_3','test_bi-encoder-{}-{}'.format(args.model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    test_retrieval(args.model_name,args.test_data_folder,corpus,args.run_dir,args.media,args.use_precomputed_corpus_embeddings)

def main():
    args=get_args()
     
    test(args)

if __name__ == "__main__":
    
    main() 