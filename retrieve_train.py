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
from retrieval.eval.eval_cross_encoder_mocheg import test_cross_encoder
from retrieval.eval.eval_msmarco_mocheg import test

from retrieval.training.train_bi_encoder_mnrl_mocheg import train
from retrieval.training.train_cross_encoder_mocheg import train_cross_encoder 

def get_args():
    
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size" , type=int)#480
    parser.add_argument("--media", type=str  ) #txt,img 
    parser.add_argument("--max_seq_length",  type=int)#100
    parser.add_argument("--model_name" )# 'clip-ViT-B-32','multi-qa-MiniLM-L6-cos-v1'
    parser.add_argument("--max_passages", default=0, type=int)
    parser.add_argument("--epochs",   type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
    parser.add_argument("--warmup_steps", default=50, type=int)#1000
    parser.add_argument("--lr", default=1e-7, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--use_pre_trained_model", default=True, action="store_false")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--ce_score_margin", default=3.0, type=float)
    parser.add_argument("--use_cached_train_queries", default=False, action="store_true")
    parser.add_argument('--train_data_folder', help='input',   default="data/train")#/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/train
    parser.add_argument('--test_data_folder', help='input',   default="data/test") 
    parser.add_argument('--val_data_folder', help='input',   default="data/val") 
    parser.add_argument("--do_val", default=True, action="store_false")
    parser.add_argument("--desc", type=str  ) #txt,img 
    parser.add_argument("--mode", type=str , default="train" ) #txt,img  
    parser.add_argument("--train_config", type=str,default="IMAGE_MODEL" )
    parser.add_argument("--use_precomputed_corpus_embeddings", default=True, action="store_false")
    parser.add_argument("--weight_decay", default=0.001, type=float) #TODO 0.01 for text
    parser.add_argument("--freeze_text_layer_num", default=17, type=int)
    parser.add_argument("--freeze_img_layer_num", default=20, type=int)
    parser.add_argument('--top_candidate_corpus_path', help='input',   default="/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00069-test_bi-encoder--home-menglong-workspace-code-misinformation_detection-retrieval-output-runs_2-train_bi-encoder-mnrl-multi-qa-MiniLM-L6-cos-v1-margin_3.0-2022-05-30_00-53-08-2022-06-03_18-53-28/query_result_txt.csv") 
     
    args = parser.parse_args()

    print(args)
    return args

def main():
    args=get_args()
    if args.mode=="train":
        if args.train_config=="CROSS_ENCODER":
            train_cross_encoder(args)
        else:
            train(args)
    elif args.mode=="test":
        if args.train_config=="CROSS_ENCODER":
            test_cross_encoder(args)
        else:
            test(args)
 
        

if __name__ == "__main__":
    
    main() 