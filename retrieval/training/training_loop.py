import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
from retrieval.search.image_search import ImageSearcher
from retrieval.search.lexical_search import LexicalSearcher
from retrieval.search.semantic_search import SemanticSearcher
from retrieval.utils.config import config
from retrieval.utils.data_util import SnopesDataset, load_snopes_data
from torch.utils.data import DataLoader
from nltk import tokenize
from itertools import zip_longest
from retrieval.utils.metrics import ImageScorer, TextScorer
from retrieval.utils.saver import Saver


def training_loop(args,rank=0):
                  #Number of passages we want to retrieve with the bi-encoder
    dataloader,relevant_document_text_list,relevant_document_img_list=load_snopes_data(args.txt_in_dir,args.relevant_document_dir)
    saver=Saver()
    if args.media=="txt":
        text_retrieve(args,relevant_document_text_list,dataloader,saver)
    elif   args.media=="img":
        image_retrieve(args,relevant_document_img_list,dataloader,saver)
    elif args.media=="img_txt":
        text_retrieve(args,relevant_document_text_list,dataloader,saver)
        image_retrieve(args,relevant_document_img_list,dataloader,saver)
 
        
    

 
def group_elements(n, iterable, padvalue=""):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)



def gen_relevant_document_text_paragraph_list(sent_num,relevant_document_text_list ):
    relevant_document_text_paragraph_list =[]
    i=0  
    for relevant_document_text in relevant_document_text_list:
        i+=1
        # print(i)
        relevant_document_text_sent_list=tokenize.sent_tokenize(relevant_document_text)
        for output in group_elements(sent_num,relevant_document_text_sent_list):
            relevant_document_text_paragraph_list.append(" ".join(output))
        
    return relevant_document_text_paragraph_list
    

def text_retrieve(args,relevant_document_text_list,dataloader,saver):
    relevant_document_text_paragraph_list=gen_relevant_document_text_paragraph_list(args.sent_num,relevant_document_text_list)
    searcher=SemanticSearcher(args.bi_encoder_checkpoint,args.cross_encoder_checkpoint,args.no_rerank)
    searcher.encode_corpus(relevant_document_text_paragraph_list)
    scorer=TextScorer(args.bi_encoder_checkpoint,args.score_by_fine_tuned)
    valid_num=1
    precision,recall=0,0
    
    length=len(dataloader)
    for iters in range(length):
        claim,text_evidence_list,truthfulness,claim_id,_,ruling_outline,ruling_article =dataloader.dataset[iters]
        
        semantic_results=searcher.search(claim,args.top_k   )
        
        if len(text_evidence_list)>0  :
            valid_num+=1
            cur_precision,cur_recall=scorer.precision_recall_by_similarity(semantic_results,relevant_document_text_paragraph_list,text_evidence_list)
            precision+=cur_precision
            recall+=cur_recall
        if  iters%100==0:
            print(f"{iters}/{length}: {precision/valid_num}, {recall/valid_num}")
        if config.verbose==True:
            print(f"claim:{claim},semantic_results:{semantic_results}")
        saver.add_retrieved_text(claim,semantic_results,relevant_document_text_paragraph_list,truthfulness,claim_id,ruling_outline,ruling_article)
    precision/=(valid_num-1)
    recall/=(valid_num-1)
    saver.save(args.csv_out_dir)
    print(f"{precision}, {recall},{compute_f1(precision, recall)}")
 
 
def compute_f1(precision, recall):
    f1=2*precision*recall/(precision+recall)
    return f1 

def gen_claim(claim):
    claim_token_list=tokenize.word_tokenize(claim)
    safe_clip_text_len=77 
    if len(claim_token_list)>safe_clip_text_len:
        claim_token_list=claim_token_list[0:safe_clip_text_len]
        # claim=" ".join(claim_token_list)
    return claim 
    
def gen_retrieved_imgs( hits,relevant_document_list):
    retrieved_document_list=[]
    for hit in hits:
        retrieved_document_list.append(relevant_document_list[hit['corpus_id']])
    
    return ";".join(retrieved_document_list)




def image_retrieve(args,relevant_document_img_list,dataloader ,saver):
    image_searcher=ImageSearcher(args.image_encoder_checkpoint)
    image_corpus=args.relevant_document_image_dir
    if args.use_precomputed_embeddings=="y":
        use_precomputed_embeddings_flag=True 
    else:
        use_precomputed_embeddings_flag=False 
    image_searcher.encode_corpus(image_corpus,relevant_document_img_list,args.relevant_document_dir,use_precomputed_embeddings_flag)
    scorer=ImageScorer(args.image_encoder_checkpoint,args.score_by_fine_tuned)
    precision,recall=0,0
    retrieved_imgs_list=[]
    valid_num=1
    length=len(dataloader)
    for iters in range(length):
        claim,text_evidence_list,truthfulness,claim_id,img_evidence_list,_,_ =dataloader.dataset[iters]
         
        # claim=gen_claim(claim)
   
        semantic_results=image_searcher.search(claim,args.top_k   )
        retrieved_imgs_list.append(gen_retrieved_imgs(semantic_results,relevant_document_img_list))
        if len(img_evidence_list)>0:
             
            valid_num+=1
            cur_precision,cur_recall=scorer.precision_recall_by_similarity(semantic_results,relevant_document_img_list,img_evidence_list,image_corpus)
            precision+=cur_precision
            recall+=cur_recall
        if iters%100==0:
            print(f"{iters}: {precision/valid_num}, {recall/valid_num}")
        if config.verbose==True:
            print(f"claim:{claim} ")
    precision/=(valid_num-1)
    recall/=(valid_num-1)
    saver.insert_and_save(args.csv_out_dir,"img_evidences",retrieved_imgs_list)
    print(f"{precision}, {recall},{compute_f1(precision, recall)}")

    
    

    
    
    
    
    

    
