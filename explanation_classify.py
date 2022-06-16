from PIL import Image
import os

import numpy as np
import click
import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from time import sleep
import warnings
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
 
from main import setup
from verification.inference import inference
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
# from controllable_generation.classification.mp_inference import mp_inference 
from controllable.classification.util.constants import * 
from controllable.classification.util.metric import *
from controllable.classification.train_verify import train_loop 

 
# @click.option('--train_txt_dir', help='input', required=True, metavar='DIR',default='verification/input/train/evidences.csv')
# @click.option('--train_img_dir', help='input', required=True, metavar='DIR',default='verification/input/train/images')
# @click.option('--test_txt_dir', help='input', required=True, metavar='DIR',default='verification/input/test/evidences.csv')
# @click.option('--test_img_dir', help='input', required=True, metavar='DIR',default='verification/input/test/images')    
    
    
def hyper_search(config_kwargs):
    config = {
        "lr":   tune.choice([0.1, 0.01, 0.0001,]), #0.1, 0.01, 0.001, 0.0001,0.00001
        "batch_size":    tune.choice([4,  256,1024])#1,2,4,8,16,32,64,128,256,512,1024,2048
        
        
    }
    gpus_per_trial = 0.25
    num_samples=8
    max_num_epochs=1
    metric_name= "test_f1"
    cpus_per_trial=3

    scheduler = ASHAScheduler(
        metric= metric_name,
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)         
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=[  "test_f1"  , "training_iteration"],max_progress_rows=num_samples)    #"reconstruct_loss",               
    result = tune.run(
        partial(train_loop ,config_kwargs=config_kwargs ),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        # stop=stopper,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False) 
        

    best_trial = result.get_best_trial(metric_name, "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final score: {}".format(
        best_trial.last_result[ metric_name]))   

@click.command()
@click.pass_context
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR',default="controllable/classification/output/runs")
@click.option('--train_dir', help='input', required=True, metavar='DIR',default='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/train') 
@click.option('--val_dir', help='input', required=True, metavar='DIR',default='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/val') 
@click.option('--test_dir', help='input', required=True, metavar='DIR',default='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test') 
@click.option('--evidence_file_name', help='input', required=True, metavar='DIR',default="Corpus2_for_controllable_generation.csv") #'retrieval_result.csv' 
@click.option('--verbos', type=str,default="y" )  
@click.option('--mode', type=str,default="train" )  
@click.option('--model_type', type=str,default="CLAIM_TEXT_IMAGE_attention_5_4" )  #CLAIM_IMAGE
@click.option('--batch_size', type=int,default=40, metavar='INT') 
@click.option('--lr', type=float,default=5e-5 )
@click.option('--early_stop', type=int,default=10, metavar='INT')
@click.option('--loss_weight_power', type=int,default=2, metavar='INT')
@click.option('--is_wandb', type=str,default="y" )  #CLAIM_IMAGE
@click.option('--pre_trained_dir', type=str,default='bert-base-uncased') 
@click.option('--concat_hidden_size', type=int,default=128) 
@click.option('--final_hidden_size', type=int,default=64) 
@click.option('--max_len', default=512, type=int, help='maximum tokens a batch')
@click.option('--max_source_length', default=512, type=int, help='maximum tokens a batch') 
@click.option('--epoch', default=100, type=int, help='force stop at specified epoch')
@click.option('--style', type=str,default='all') 
@click.option('--freeze_bert_layer_number', default=20, type=int )#16 parameters per layer
@click.option('--accum_iter', default=1, type=int, help='grad accum step')
#for inference
@click.option('--checkpoint_dir', help='input',  metavar='DIR')  #,default='verification/output/runs/00121-'
@click.option('--save_predict', type=str,default="y" ) 
# @click.option('--final_checkpoint_dir', help='dir for final paper results', required=True, metavar='DIR',default='verification/output/models/text_image') 
def main(ctx,  **config_kwargs):
 
    mode=config_kwargs["mode"]
    if mode =="train":
        train_loop(None,config_kwargs)
    elif mode=="hyper_search":
        hyper_search(config_kwargs)
    # elif mode=="mp_test":
    #     mp_inference(config_kwargs)
    else:
        train_loop(None,config_kwargs)
    
     




if __name__ == "__main__":
    
    main() 