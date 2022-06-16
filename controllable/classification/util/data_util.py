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
    
    def __init__(self, ruling_outline_data,  labels,   tokenizer,max_len ):
        self.ruling_outline_data = ruling_outline_data #one text
        # self.evidence_data = evidence_data  #list of text
      
        self.labels = labels  #int
     
        self.tokenizer=tokenizer 
        self.max_len=max_len

    def __getitem__(self, idx):
        sample = {} 
        ruling_outline_data = self.ruling_outline_data[idx]
        
        # claim_and_text_evidence_list=get_claim_and_text_evidence_list(claim,text_evidence_list)
        label = self.labels[idx]
        # image_list_str=self.images[idx]
        # img_list=[]
        # if not pd.isna(image_list_str):
        #     images_path_list = image_list_str.split(';')
        #     img_list=open_image(images_path_list,self.img_dir  )
            
        # if img_list!= []:
        #     # processor for clip model    
        #     inputs = self.processor(text=claim_and_text_evidence_list, images=img_list, return_tensors="pt", padding=True,truncation=True)
        # else:
        #     inputs = self.processor(text=claim_and_text_evidence_list,  return_tensors="pt", padding=True,truncation=True)
             
        claim_evidence_tokens = self.tokenizer(ruling_outline_data,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               max_length=self.max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length" )

        
        
        return claim_evidence_tokens,torch.tensor(label)
    
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

def get_data_loader(train_data_dir, val_data_dir, test_data_dir, style,  tokenizer,max_len,batch_size):
    test_cleaned_ruling_outline_data, test_truth_claim_evidence_data  , test_labels,test_label_statistic = get_data(test_data_dir, style)
    val_cleaned_ruling_outline_data, val_truth_claim_evidence_data  , val_labels,val_label_statistic= get_data(val_data_dir, style)
    train_cleaned_ruling_outline_data, train_truth_claim_evidence_data  , train_labels,train_label_statistic = get_data(train_data_dir, style)
    train_dataset =  MisinformationDataset(train_cleaned_ruling_outline_data ,train_labels,  tokenizer,max_len )
    val_dataset =  MisinformationDataset(val_cleaned_ruling_outline_data, val_labels,  tokenizer,max_len )
    test_dataset = MisinformationDataset(test_cleaned_ruling_outline_data , test_labels,  tokenizer ,max_len )
    
   
    ## We call the dataloader class
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size//2,
        shuffle=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size//2,
        shuffle=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size//2,
        shuffle=True,
        drop_last=True
    )

    dataloaders = {'Train': train_loader, 'Test': test_loader, 'Val': val_loader}
    return dataloaders,train_label_statistic

def get_data(data_file, style):
    cleaned_ruling_outline_data = []
    truth_claim_evidence_data = []
    labels = []
    evidences=[]
    refuted=0
    supported=0
    nei=0
    df_data = pd.read_csv(data_file)
    for _,row in df_data.iterrows():
        true_label = row['cleaned_truthfulness']
        truth_claim_evidence = row['truth_claim_evidence']
        cleaned_ruling_outline = row['cleaned_ruling_outline']
        # evidence=row['Evidence']
        
        if style!=None and style!="all" and true_label!=style:
            continue 
        else:
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
                print(f"error {true_label}")
                exit()        
        cleaned_ruling_outline_data.append(cleaned_ruling_outline.strip())
        truth_claim_evidence_data.append(truth_claim_evidence.strip())  
        # evidences.append(evidence)
    return cleaned_ruling_outline_data, truth_claim_evidence_data , labels,[refuted,supported,nei]
            