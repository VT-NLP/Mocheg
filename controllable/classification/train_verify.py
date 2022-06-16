from email.policy import strict
from PIL import Image
import os
import wandb
import numpy as np

import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from time import sleep
import warnings
import torch.nn.functional as F
from main import setup
from ray import tune
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel, AdamW
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.read_example import get_relevant_document_dir

from controllable.classification.util.data_util import get_data_loader, get_loss_weights
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
from controllable.classification.util.constants import *
from controllable.classification.model import   TextModel
from controllable.classification.util.enums import ModelAttribute 
from controllable.classification.util.metric import *   
import retrieval.utils as utils
  
def verify_init():
    warnings.filterwarnings("ignore") 

    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled   = True
    return device
  
def freeze(model):
    model.requires_grad_(False)
    # for param in model.parameters():
    #     param.requires_grad = False         
def   update_config(config_kwargs,config):
    if config!=None:
       
        if "batch_size" in config.keys():
            config_kwargs['batch_size']=config["batch_size"]
        if "lr" in config.keys():
            config_kwargs['lr']=config["lr"]
    return config_kwargs

def setup_args_for_verify(config_kwargs):
    args,logger = setup(config_kwargs)   
    args.train_txt_dir=os.path.join(args.train_dir,args.evidence_file_name)
    args.val_txt_dir=os.path.join(args.val_dir,args.evidence_file_name)
    args.test_txt_dir=os.path.join(args.test_dir,args.evidence_file_name )
    whole_dataset_dir=get_relevant_document_dir(args.test_dir)
    args.train_img_dir=os.path.join(whole_dataset_dir,"images")
    args.val_img_dir=os.path.join(whole_dataset_dir,"images") 
    args.test_img_dir=os.path.join(whole_dataset_dir,"images")
    return args,logger


def classifier_predict(tgt_text_list,max_len,device,classifier_tokenizer,classifier):
    tgt_idx=classifier_tokenizer(tgt_text_list,add_special_tokens=True,  
                                               max_length=max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length").to(device)
    # tgt_idx = collate_fn(tgt).to(device)
    tgt_cls = F.softmax(classifier(**tgt_idx).detach(),-1)
    return tgt_cls

def load_classifier(checkpoint_dir ):
    device=torch.device("cuda")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    model = TextModel(bert_model,  128, 64, 3,20).to(device)
    path=os.path.join(checkpoint_dir,"base.pt")
    model.load_state_dict(torch.load(path)  )#
    print(f"resume from {checkpoint_dir}")
    model.requires_grad_(False)
    classifier_tokenizer=  BertTokenizer.from_pretrained("bert-base-uncased")
    classifier_tokenizer.truncation_side="left"
    return model,classifier_tokenizer

def train_loop( tuner_config,config_kwargs):
    config_kwargs=update_config(config_kwargs,tuner_config)
    args,logger=setup_args_for_verify(config_kwargs)
    num_classes=3
    
    device=verify_init()
    # model_type_enum=ModelAttribute[args.model_type]
  
    tokenizer = BertTokenizer.from_pretrained(args.pre_trained_dir)
    tokenizer.truncation_side="left"
    dataloaders,train_label_statistic=get_data_loader(args.train_txt_dir, args.val_txt_dir,args.test_txt_dir,args.style,
                                                      tokenizer,args.max_len,args.batch_size)
    # model = MultiModal2(bert = None, clip = clip_model, concat_hidden_size = 256, final_hidden_size = 64, 
    #                    num_classes = 3,max_image_num=args.max_image_num,model_type=args.model_type,model_type_enum=model_type_enum ,
    #                    is_project_embed=model_type_enum.is_project_embed,only_claim_setting=model_type_enum.only_claim_setting,
    #                    token_pooling_method=model_type_enum.token_pooling_method).to(device)

    
    bert_model = BertModel.from_pretrained(args.pre_trained_dir)
    model = TextModel(bert_model, args.concat_hidden_size, args.final_hidden_size, num_classes,args.freeze_bert_layer_number).to(device)
    # freeze(clip_model)
    # model=load_model(opt,"train")
    model=resume_model(args.checkpoint_dir,model)
    
    print(model)
    #optimizer
    optimizer = Adam(model.parameters(), lr = args.lr, eps=1e-8 )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=6000, 
        num_training_steps=len(dataloaders["Train"])*args.epoch
    )
    train(model,dataloaders,device,optimizer, args,train_label_statistic,logger,args.batch_size,scheduler,args.accum_iter)

def resume_model(checkpoint_dir,model):
    if checkpoint_dir!=None:
        path=os.path.join(checkpoint_dir,"base.pt")
        model.load_state_dict(torch.load(path),strict=False )#
        print(f"resume from {checkpoint_dir}")
    
    return model   

def train_phase(accum_iter,model,dataloader,device,optimizer ,criterion,scheduler):
    print("Train" + ":")
    live_loss = 0
    model.train()
    # wandb.watch(model.verifier.stance_detect_layer, log='all')
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            claim_img_encod,labels=batch
            labels=labels.to(device)
            input_ids = claim_img_encod['input_ids'].to(device).squeeze(1)
            attention_mask = claim_img_encod['attention_mask'].to(device).squeeze(1)
            token_type_ids = claim_img_encod['token_type_ids'].to(device).squeeze(1)
            output = model( input_ids,attention_mask,token_type_ids )
            loss = criterion(output, labels)
            live_loss+=loss
            # normalize loss to account for batch accumulation
            loss = loss / accum_iter
            # Backward pass  (calculates the gradients)
            loss.backward()
            # weights update
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                # gradient clipping
                # nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_VALUE)
                # wandb.log({"live_loss":loss})
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()     
                
            avg_loss= live_loss.item()/(batch_idx+1)
            
            
            
            tepoch.set_postfix(loss=avg_loss)
    wandb.log({"loss": avg_loss,"live_loss":loss})
    return avg_loss


def val_phase(accum_iter,model,dataloader,device, best_valid_f1,args,is_save_predict,epoch,best_epoch  ):
    print("Val" + ":")
    y_true = []
    y_pred = []
    claim_id_list=[]
    model.eval()
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            claim_img_encod,labels=batch
            input_ids = claim_img_encod['input_ids'].to(device).squeeze(1)
            attention_mask = claim_img_encod['attention_mask'].to(device).squeeze(1)
            token_type_ids = claim_img_encod['token_type_ids'].to(device).squeeze(1)
            output = model( input_ids,attention_mask,token_type_ids )
            labels_numpy = labels.numpy()
            _, preds = output.data.max(1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels_numpy.tolist())
           
            # live_acc = get_accuracy(y_pred, y_true)
            tepoch.set_postfix( )
    f1,pre,recall=calc_metric( y_true, y_pred,args )
    wandb.log({"f1": f1,"epoch":epoch})
    best_valid_f1,best_epoch=save_model(best_valid_f1,f1,epoch,best_epoch,model,args)
     
    if args.early_stop and (epoch - best_epoch) >= args.early_stop:
        print('early stop at epc {}'.format(epoch))
        is_early_stop=True
    else:
        is_early_stop=False
    return f1,f1,pre,recall,best_valid_f1,best_epoch,is_early_stop

def save_prediction(claim_id_list,y_pred,args):
    prediction={"claim_id":claim_id_list,"cleaned_truthfulness":y_pred}
    df = pd.DataFrame(prediction)
    if args.mode in["mp_test","inference"]:
        run_dir=args.checkpoint_dir
    else:
        run_dir=args.test_dir
    
    if "retrieval" not in args.evidence_file_name :
         
        prediction_path=os.path.join(run_dir,"verification_result.csv" )
        
    else:
        prediction_path=os.path.join(run_dir,"verification_result_after_retrieval.csv")
    print(prediction_path)
    df.to_csv(prediction_path,index=False)

def train(model,dataloaders,device,optimizer,args,train_label_statistic,logger,batch_size,scheduler,accum_iter):
    best_valid_f1 = 0
    best_epoch=0
    #Loss function
    normed_weights=get_loss_weights(device,train_label_statistic,args.loss_weight_power)
    logger.write(f"normed:{normed_weights}")
    criterion = nn.CrossEntropyLoss(weight=normed_weights)
    # batch accumulation parameter
    #4
    if args.is_wandb=="y":
        wandb.init(project=f'classification-{args.mode}',config=args)
        wandb.run.name=f"{args.cur_run_id}-{wandb.run.name}"
    epoch=0
    if args.mode in ["train","hyper_search"]:
        print('Epoch {}/{}'.format(epoch, args.epoch))
        epoch_avg_acc,f1,pre,recall,best_valid_f1,best_epoch,is_early_stop=val_phase(accum_iter, model,dataloaders["Val"],device,best_valid_f1 ,args,"n",epoch ,best_epoch)
        logger.write("F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}, Accuracy: {:.4f}".format(f1, pre, recall, epoch_avg_acc))
        for epoch in range(0, args.epoch):
            print('-'*50)
            print('Epoch {}/{}'.format(epoch+1, args.epoch))   
            epoch_avg_loss=train_phase(accum_iter,model,dataloaders["Train"],device,optimizer ,criterion,scheduler)  
            epoch_avg_acc,f1,pre,recall,best_valid_f1,best_epoch,is_early_stop=val_phase(accum_iter, model,dataloaders["Val"],device,best_valid_f1 ,args,"n",epoch ,best_epoch)
            c="F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}, Accuracy: {:.4f}, Loss: {:.4f}.".format(f1, pre, recall, epoch_avg_acc, epoch_avg_loss)
            logger.write(c)
            if is_early_stop:
                break
    test_phase(accum_iter,model,dataloaders,device, args, logger,epoch+1,epoch+1)
    wandb.finish()
    
def test_phase(accum_iter,model,dataloaders,device, args, logger,epoch,best_epoch):
    print('Final test ' )
    if args.mode in ["train","hyper_search"]:
        model=resume_model( args.run_dir ,model)
    epoch_avg_acc,f1_score,pre,recall,best_valid_f1,best_epoch,is_early_stop=val_phase( accum_iter,model,dataloaders["Test"],device,np.inf ,args,args.save_predict ,epoch,best_epoch)
    wandb.log({"test_f1": f1_score })
    logger.write("F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}, Accuracy: {:.4f}".format(f1_score, pre, recall, epoch_avg_acc))
    if args.mode=="hyper_search": 
        tune.report(test_f1=f1_score)
                    
    
def calc_metric( y_true, y_pred,args  ):
    
    # pre = precision_score(y_true, y_pred, average='micro')
    # recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    pre=f1
    recall=f1
    print(f"f1 {f1}")
    
    if args.verbos=="y":
        # print(y_pred)
        confusion_matrix_result=confusion_matrix(y_true, y_pred)
        print(confusion_matrix_result)
        c_report=classification_report(y_true, y_pred, target_names =  ['Refuted', 'Supported', 'NEI'],output_dict=True)
        print(c_report)
        
        wandb.log({  "by_label/refuted_f1":c_report["Refuted"]["f1-score"],"by_label/supported_f1":c_report["Supported"]["f1-score"],"by_label/nei_f1":c_report["NEI"]["f1-score"]})
      
        
    
    print()   
    return f1,pre,recall
    
def save_model(best_valid_f1,f1,epoch,best_epoch,model,args):
    if f1 > best_valid_f1:
        best_valid_f1 = f1
        best_epoch=epoch
        torch.save(model.state_dict(), os.path.join(args.run_dir,"base.pt") )
        print('Model Saved!')
    return best_valid_f1,best_epoch