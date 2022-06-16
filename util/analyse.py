import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
from retrieval.search.image_search import ImageSearcher
from retrieval.search.lexical_search import LexicalSearcher
from retrieval.search.semantic_search import SemanticSearcher
from retrieval.training.training_loop import gen_relevant_document_text_paragraph_list
from retrieval.utils.config import config
from retrieval.utils.data_util import SnopesDataset, load_snopes_data
from torch.utils.data import DataLoader
from nltk import tokenize
from itertools import zip_longest
from retrieval.utils.metrics import ImageScorer, TextScorer
from retrieval.utils.saver import Saver
from util.read_example import get_relevant_document_dir


def training_loop(args,rank=0):
                  #Number of passages we want to retrieve with the bi-encoder
    args.sent_num=1
    args.txt_in_dir=args.in_dir
    args.img_in_dir=os.path.join(args.in_dir,"images")
    if args.csv_out_dir is None:
        args.csv_out_dir=os.path.join(args.in_dir,"retrieval/retrieval_result.csv")
    args.relevant_document_dir=get_relevant_document_dir(args.txt_in_dir)
    args.relevant_document_image_dir=os.path.join(args.relevant_document_dir,"images")
    dataloader,relevant_document_text_list,relevant_document_img_list=load_snopes_data(args.txt_in_dir,args.relevant_document_dir)
    analyse_text(args,relevant_document_text_list,dataloader)
    

def analyse_text(args,relevant_document_text_list,dataloader,saver):
    relevant_document_text_paragraph_list=gen_relevant_document_text_paragraph_list(args.sent_num,relevant_document_text_list)
    relevant_document_len_sum=0
    evidence_num=0
    relevant_document_num=len(relevant_document_text_paragraph_list)
    for relevant_document_text_paragraph in relevant_document_text_paragraph_list:
        relevant_document_len_sum+=len(tokenize.word_tokenize(relevant_document_text_paragraph))
    valid_num=1
    evidence_len_sum=0
    
    length=len(dataloader)
    for iters in range(length):
        claim,text_evidence_list,truthfulness,claim_id,_,ruling_outline,ruling_article =dataloader.dataset[iters]
        
        if len(text_evidence_list)>0  :
            valid_num+=1
            evidence_num+=len(text_evidence_list)
            for text_evidence in text_evidence_list:
                evidence_len_sum+=len(tokenize.word_tokenize(text_evidence))
        if  iters%100==0:
            print(f"{iters}/{length} ")
         
         
    evidence_len_per_claim=evidence_len_sum/(valid_num)
   
    print(f"{evidence_len_sum},{valid_num}, {evidence_len_per_claim},{relevant_document_len_sum},{evidence_num/valid_num},{relevant_document_num}") 
 
 