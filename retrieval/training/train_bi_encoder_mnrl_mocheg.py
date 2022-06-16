"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.
For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)
Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.
With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus
Running this script:
python train_bi-encoder-v3.py
"""
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
from controllable.classification.model import show_model
from retrieval.training.model import MultiMediaSentenceTransformer
from retrieval.utils.enums import TrainAttribute
from util.common_util import setup_with_args
from retrieval.data.data_dealer import ImageDataDealer, MSMARCODataset, MSMARCOImageDataset, TextDataDealer, get_train_queries
from retrieval.eval.eval_msmarco_mocheg import gen_evaluator,   load_qrels, load_queries, test_retrieval

from retrieval.utils.data_util import SnopesDataset, SnopesForRetrievalDataset 
 

def load_model(args,model_name,max_seq_length):
    # Load our embedding model
    if args.use_pre_trained_model:
        logging.info("use pretrained SBERT model")
        model = MultiMediaSentenceTransformer(model_name)
        model.max_seq_length = max_seq_length
        # if args.media=="txt":
        #     model.max_seq_length = max_seq_length
    else:
        logging.info("Create new SBERT model")
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model 

def load_data( args,ce_score_margin,num_negs_per_system):
    data_folder = args.train_data_folder
    # os.makedirs(data_folder, exist_ok=True)
    # dataset=SnopesForRetrievalDataset( data_folder,"text_evidence")
    if args.media=="txt":
        data_dealer=TextDataDealer()
    else:
        data_dealer=ImageDataDealer()
    corpus=data_dealer.load_corpus(data_folder, 0)
    positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(data_folder,args.media)
    queries=data_dealer.load_queries(data_folder,needed_qids)
    train_queries=get_train_queries(queries,positive_rel_docs,negative_rel_docs)
    # ce_scores=load_ce_score_to_prevent_false_negative(data_folder)
    # train_queries=form_train_queries_with_hard_negative(data_folder,args,ce_scores,ce_score_margin,num_negs_per_system,queries)
    if args.media=="txt":
        train_dataset = MSMARCODataset(train_queries ,corpus)
    else:
        train_dataset = MSMARCOImageDataset(train_queries ,corpus)
    logging.info("Train queries: {}".format(len(train_dataset)))
    return train_dataset ,corpus



def freeze_part_bert(model,freeze_text_layer_num,freeze_img_layer_num):
    count = 0
 
    for p in model[0].model.text_model.named_parameters():
        
        if (count<=freeze_text_layer_num):
            p[1].requires_grad=False    
        else:
            break
              
        count=count+1
        print(p[0], p[1].requires_grad)
        
    count=0
    for p in model[0].model.vision_model.named_parameters():
        
        if (count<=freeze_img_layer_num):
            p[1].requires_grad=False    
        else:
            break
              
        count=count+1
        print(p[0], p[1].requires_grad)
        
    return model 

def train(args):
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
    
    # The  model we want to fine-tune
    model_name =args.model_name #'distilbert-base-uncased'
    train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
    max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
    ce_score_margin = args.ce_score_margin             #Margin for the CrossEncoder score between negative and positive passages
    num_negs_per_system = args.num_negs_per_system         # We used different systems to mine hard negatives. Number of hard negatives to add from each system
    num_epochs = args.epochs                 # Number of epochs we want to train

    # model_save_path = 'retrieval/output/runs_2/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    model_save_path,args=setup_with_args(args,'retrieval/output/runs_3','train_bi-encoder-{}-{}'.format(model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    
    model=load_model(args,model_name,max_seq_length)
    if args.media=="img":
        model=freeze_part_bert(model,args.freeze_text_layer_num,args.freeze_img_layer_num)
    # show_model(model)
    train_dataset,corpus =load_data( args,ce_score_margin,num_negs_per_system)

    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    checkpoint_path=os.path.join(model_save_path,"models")
    if args.do_val:
        val_evaluator=gen_evaluator(args.val_data_folder,corpus,args.media,model,args.use_precomputed_corpus_embeddings,model_save_path)
        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=val_evaluator,
                evaluation_steps=len(train_dataloader),
                epochs=num_epochs,
                warmup_steps=args.warmup_steps,
                use_amp=True,
                checkpoint_path=checkpoint_path,
                checkpoint_save_steps=len(train_dataloader),
                optimizer_params = {'lr': args.lr},
                output_path =model_save_path,
                weight_decay=args.weight_decay,
                media=args.media
                )
    else:
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=args.warmup_steps,
                use_amp=True,
                checkpoint_path=checkpoint_path,
                checkpoint_save_steps=len(train_dataloader)*(num_epochs/20),
                optimizer_params = {'lr': args.lr},
                output_path =model_save_path,
                weight_decay=args.weight_decay,
                media=args.media
                )  

    # Save the model
    model.save(checkpoint_path)
    test_retrieval(checkpoint_path,args.test_data_folder, corpus,model_save_path,args.media,args.use_precomputed_corpus_embeddings)
