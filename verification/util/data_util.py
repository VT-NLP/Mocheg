import torch
from PIL import Image
import os

import numpy as np

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

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from verification.util.constants import * 
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
 
import nltk 
 

def open_image(images_path,img_dir):
    img_list=[]
    for image in images_path:
        img = Image.open(os.path.join(img_dir, image))
        img_list.append(img)
    return img_list
class MisinformationDataset(torch.utils.data.Dataset):
    """
    This is our custom dataset class which will load the text and their corresponding labels into Pytorch tensors
    """
    
    def __init__(self, claim_data, evidence_data, images, labels,img_dir , processor, tokenizer ,claim_ids):
        self.claim_data = claim_data #one text
        self.evidence_data = evidence_data  #list of text
        self.images = images  #list of images
        self.labels = labels  #int
        self.img_dir=img_dir  
        self.claim_ids=claim_ids
        self.processor=processor
        self.tokenizer=tokenizer 

    def __getitem__(self, idx):
        sample = {}
        claim = self.claim_data[idx]
        text_evidence_list = self.evidence_data[idx]
        claim_and_text_evidence_list=get_claim_and_text_evidence_list(claim,text_evidence_list)
        label = self.labels[idx]
        image_list_str=self.images[idx]
        img_list=[]
        if not pd.isna(image_list_str):
            images_path_list = image_list_str.split(';')
            img_list=open_image(images_path_list,self.img_dir  )
            
        if img_list!= []:
            # processor for clip model    
            inputs = self.processor(text=claim_and_text_evidence_list, images=img_list, return_tensors="pt", padding=True,truncation=True, max_length=77)
        else:
            inputs = self.processor(text=claim_and_text_evidence_list,  return_tensors="pt", padding=True,truncation=True, max_length=77)
             
         

        try:
            sample["claim_id"] = torch.tensor(self.claim_ids[idx])
            sample["label"] = torch.tensor(label)
            sample["claim_img_encod"] = inputs
        except Exception as e:
            print(e)
        
        return sample
    
    def __len__(self):
        return len(self.labels)
        

def get_claim_and_text_evidence_list(claim,text_evidence_list):
    claim_and_text_evidence_list=[]
    claim_and_text_evidence_list.append(claim)
    claim_and_text_evidence_list.extend(text_evidence_list)
    return claim_and_text_evidence_list
 
 
def get_loss_weights(device,train_label_statistic,power=1):
    normed_weights=get_loss_weight_fun1(3,np.array(train_label_statistic),device,power)
    
    return normed_weights 

def get_loss_weight_fun2( samples_per_class ):
    normed_weights = torch.tensor([1/samples_per_class[0], 1/samples_per_class[1], 1/samples_per_class[0]])*1000
    return normed_weights
    
    
def get_loss_weight_fun1(num_of_classes,samples_per_class,device,power=1):
    weights_for_samples=1.0/np.array(np.power(samples_per_class,power))
    weights_for_samples=weights_for_samples/np.sum(weights_for_samples)*num_of_classes
    weights_for_samples=torch.tensor(weights_for_samples, dtype=torch.float32,device=device)
    return weights_for_samples

def get_data_loader(train_data_dir,train_img_dir,val_data_dir, val_img_dir,test_data_dir, test_img_dir,processor, tokenizer):
    test_ids, test_claims, test_evidences, test_images, test_labels,test_label_statistic = get_data(test_data_dir, test_img_dir)
    val_ids, val_claims, val_evidences, val_images, val_labels,val_label_statistic = get_data(val_data_dir, val_img_dir)
    train_ids, train_claims, train_evidences, train_images, train_labels,train_label_statistic = get_data(train_data_dir, train_img_dir)
    train_dataset =  MisinformationDataset(train_claims, train_evidences, train_images, train_labels,train_img_dir, processor, tokenizer,train_ids)
    val_dataset =  MisinformationDataset(val_claims, val_evidences, val_images, val_labels,val_img_dir, processor, tokenizer,val_ids)
    test_dataset = MisinformationDataset(test_claims, test_evidences, test_images, test_labels,test_img_dir, processor, tokenizer ,test_ids)
    
   
    ## We call the dataloader class
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=1,
        shuffle=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=1,
        shuffle=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=1,
        shuffle=True,
        drop_last=True
    )

    dataloaders = {'Train': train_loader, 'Test': test_loader, 'Val': val_loader}
    return dataloaders,train_label_statistic

def get_data(data_file, img_files):
    claim_id = []
    claim_data = []
    evidence_data = []
    labels = []
    images  = []
    refuted=0
    supported=0
    nei=0
    df_data = pd.read_csv(data_file)
    for _,row in df_data.iterrows():
        true_label = row['cleaned_truthfulness']
        claim = row['Claim']
        evidence = row['Evidence']
        image = row['img_evidences']
        id = row['claim_id']
        if true_label =="refuted":
            refuted+=1
            labels.append(0)
        elif true_label =="supported":    
            supported+=1
            labels.append(1)  
        elif true_label =="NEI":
            nei+=1
            labels.append(2)  
        else:
            continue        
        claim_id.append(id)
        claim_data.append(claim)
        evidence_data.append(nltk.tokenize.sent_tokenize(evidence) )
        images.append(image)
            
    return claim_id, claim_data, evidence_data, images, labels,[refuted,supported,nei]
            