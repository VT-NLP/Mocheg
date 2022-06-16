import os
import click
import re
import json
import tempfile
import torch
from main import setup
import retrieval.utils as utils
from util.read_example import get_relevant_document_dir
from retrieval.training import training_loop
from retrieval.utils.config import * 
from functools import partial
import numpy as np

#----------------------------------------------------------------------------



@click.command()
@click.pass_context
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR',default="retrieval/output/runs")
@click.option('--csv_out_dir', help='Where to save the results',   metavar='DIR' )
@click.option('--in_dir', help='input', required=True, metavar='DIR',default="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test")
# @click.option('--in_dir', help='input', required=True, metavar='DIR',default="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mode3_latest_v2")
@click.option('--top_k', help='top_k', type=int,default=5, metavar='INT')#TODO 10
@click.option('--metric', type=str,default="similarity" )  
@click.option('--sent_num',   type=int,default=1, metavar='INT')
@click.option('--media', type=str,default="img_txt" ) #txt,img_txt
@click.option('--use_precomputed_embeddings', type=str,default="y" )  # #image_search use precomputed_embeddings for images
@click.option('--bi_encoder_checkpoint',  metavar='DIR',default="/home/menglong/workspace/code/Mocheg/checkpoint/text_retrieval/bi_encoder")#"/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_2/train_bi-encoder-mnrl-multi-qa-MiniLM-L6-cos-v1-margin_3.0-2022-05-30_00-53-08"
@click.option('--cross_encoder_checkpoint',  metavar='DIR',default="cross-encoder/ms-marco-MiniLM-L-6-v2")#'/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00063-train_cross-encoder-cross-encoder-ms-marco-MiniLM-L-6-v2-2022-06-02_21-59-58-latest'
@click.option('--image_encoder_checkpoint',  metavar='DIR',default="/home/menglong/workspace/code/Mocheg/checkpoint/image_retrieval")#retrieval/output/runs_3/00081-train_bi-encoder-clip-ViT-B-32-2022-06-07_07-42-24/models
@click.option('--no_rerank',  is_flag=True, show_default=True, default=False   ) #txt,img_txt
@click.option('--score_by_fine_tuned',  is_flag=True,  default=True   ) #txt,img_txt
def main(ctx,  **config_kwargs):
    args,logger=setup(config_kwargs)
    args.txt_in_dir=args.in_dir
    args.img_in_dir=os.path.join(args.in_dir,"images")
    if args.csv_out_dir is None:
        args.csv_out_dir=os.path.join(args.in_dir,"retrieval/retrieval_result.csv")
    args.relevant_document_dir=get_relevant_document_dir(args.txt_in_dir)
    args.relevant_document_image_dir=os.path.join(args.relevant_document_dir,"images")
    training_loop.training_loop(args,rank=0)
  

if __name__ == "__main__":
    
    main() 