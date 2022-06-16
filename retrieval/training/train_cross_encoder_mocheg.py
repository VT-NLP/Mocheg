"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.
The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.
This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.
Running this script:
python train_cross-encoder.py
"""
import torch
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from torch import nn
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
from retrieval.data.data_dealer import TextDataDealer, get_queries_with_content, get_train_queries, load_cross_encoder_samples
from retrieval.eval.eval_msmarco_mocheg import load_qrels
from retrieval.utils.enums import TrainAttribute

from util.common_util import setup_with_args
from verification.util.data_util import get_loss_weight_fun1, get_loss_weights



def set_up():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout
    
    
def load_train_samples(sub_data_folder,args,data_dealer,corpus,num_max_negatives):
    positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(sub_data_folder,args.media)
    queries=data_dealer.load_queries(sub_data_folder,needed_qids)
    train_samples,pos_weight=load_cross_encoder_samples(sub_data_folder,queries,corpus ,num_max_negatives)
    return train_samples,pos_weight
    
def load_val_query_dic(sub_data_folder, args,data_dealer,corpus):
    positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(sub_data_folder,args.media)
    queries=data_dealer.load_queries(sub_data_folder,needed_qids)
    queries_dict=get_queries_with_content(queries,positive_rel_docs,negative_rel_docs,corpus)
    return  queries_dict
        
def load_data(data_folder,pos_neg_ration,max_train_samples,args):
    data_dealer=TextDataDealer()
    corpus=data_dealer.load_corpus(data_folder, 0) 
    num_max_negatives=20
    train_samples,pos_weight=load_train_samples(data_folder, args,data_dealer,corpus,num_max_negatives)
    dev_queries_dict=load_val_query_dic(args.val_data_folder, args,data_dealer,corpus)
    
     
 
  
            
    return train_samples,dev_queries_dict,pos_weight
def train_cross_encoder(args):
    #First, we define the transformer model we want to fine-tune
    train_attribute=TrainAttribute[args.train_config]
    model_name =train_attribute.model_name
    train_batch_size = train_attribute.batch_size
    num_epochs =train_attribute.epoch
    args.media=train_attribute.media 
    model_save_path,args=setup_with_args(args,'retrieval/output/runs_3','train_cross-encoder-{}-{}'.format(model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    # model_save_path = 'output/training_ms-marco_cross-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # We train the network with as a binary label task
    # Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
    # We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
    # in our training setup. For the negative samples, we use the triplets provided by MS Marco that
    # specify (query, positive sample, negative sample).
    pos_neg_ration = 4
    # Maximal number of training samples we want to use
    max_train_samples = 2e7
    ### Now we read the MS Marco dataset
    data_folder = args.train_data_folder
    # os.makedirs(data_folder, exist_ok=True)
    
    #We set num_labels=1, which predicts a continous score between 0 and 1
    model = CrossEncoder(model_name, num_labels=1, max_length=train_attribute.max_seq_length)
    train_samples,dev_samples,pos_weight=load_data(data_folder,pos_neg_ration,max_train_samples,args)

    # We create a DataLoader to load our train samples
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    # We add an evaluator, which evaluates the performance during training
    # It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
    evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

    # Configure the training
    warmup_steps = 5000
    logging.info("Warmup-steps: {}".format(warmup_steps))
    device=torch.device("cuda")
     
    logging.info(f"normed:{pos_weight}")
    loss_fct= nn.BCEWithLogitsLoss(pos_weight= torch.tensor(pos_weight) )
    # Train the model
    model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_params = {'lr': args.lr},
            use_amp=True,
            loss_fct=loss_fct)

    #Save latest model
    model.save(model_save_path+'-latest')