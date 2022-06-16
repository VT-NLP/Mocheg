import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm
from sentence_transformers import InputExample

from util.read_example import read_image, read_news, read_relevant_document
torch.set_num_threads(4)
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import pandas as pd
import torch
from torch.utils.data import Dataset 
from retrieval.utils.news import News
from torch.utils.data import DataLoader

def download_image(img_dir):
    # Next, we get about 25k images from Unsplash 
 
    if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
        os.makedirs(img_dir, exist_ok=True)
        
        photo_filename = 'unsplash-25k-photos.zip'
        photo_dir="~/workspace/data/unsplash/"+photo_filename
        if not os.path.exists(photo_dir):   #Download dataset if does not exist
            util.http_get('http://sbert.net/datasets/'+photo_filename,photo_dir )
            
        #Extract all images
        with zipfile.ZipFile(photo_dir, 'r') as zf:
            for member in tqdm(zf.infolist(), desc='Extracting'):
                zf.extract(member, img_dir)



class SnopesForRetrievalDataset(Dataset):
    def __init__(self, txt_in_dir,content_type="text_evidence",skip_empty_text_evidence=True,paragraph_sent_num=1):
        self.news_dict,self.relevant_document_text_list,self.relevant_document_img_list=read_news(txt_in_dir,content_type,skip_empty_text_evidence,paragraph_sent_num)
        

    def __len__(self):
        return len(self.news_dict)

    def __getitem__(self, idx):
        
        keys_list = list(self.news_dict)
        key = keys_list[idx]
        news=self.news_dict[key]
        claim=news.claim
        # text_evidence_list=
        truthfulness=news.truthfulness
        claim_id=news.claim_id
        img_evidence_list=news.get_img_evidence_list()
        ruling_outline=news.ruling_outline
        ruling_article=news.ruling_article

        query_text=claim 
        pos_text=news.get_text_evidence_list().pop(0)
        news.get_text_evidence_list().append(pos_text)
        # neg_text=news.ruling_outline
        
        return InputExample(texts=[query_text, pos_text])#, neg_text
 
 

     

class SnopesDataset(Dataset):
    def __init__(self, txt_in_dir,content_type="all"):
         
        self.news_dict,self.relevant_document_text_list,self.relevant_document_img_list=read_news(txt_in_dir,content_type)
        

    def __len__(self):
        return len(self.news_dict)

    def __getitem__(self, idx):
        
        keys_list = list(self.news_dict)
        key = keys_list[idx]
        news=self.news_dict[key]
        claim=news.claim
        text_evidence_list=news.get_text_evidence_list()
        truthfulness=news.truthfulness
        claim_id=news.claim_id
        img_evidence_list=news.get_img_evidence_list()
        ruling_outline=news.ruling_outline
        ruling_article=news.ruling_article
 
        return claim, text_evidence_list,truthfulness,claim_id,img_evidence_list,ruling_outline,ruling_article

    def get_relevant_document_text_list(self):
        return self.relevant_document_text_list

    def get_relevant_document_img_list(self):
        return self.relevant_document_img_list
        
 
def get_relevant_document(txt_in_dir,relevant_document_dir): 
    news_dict={}
    news_dict,relevant_document_img_list=read_image(relevant_document_dir,news_dict,content="img")
    news_dict,relevant_document_text_list=read_relevant_document(txt_in_dir,news_dict)
    # dataset=SnopesDataset( relevant_document_dir)
    # relevant_document_text_list=dataset.get_relevant_document_text_list()
    # relevant_document_img_list=dataset.get_relevant_document_img_list()
    return relevant_document_text_list,relevant_document_img_list
 
 
def load_snopes_data(txt_in_dir, relevant_document_dir):
    dataset=SnopesDataset( txt_in_dir,content_type="evidence")
    relevant_document_text_list,relevant_document_img_list=get_relevant_document(txt_in_dir,relevant_document_dir)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  
    return train_dataloader,relevant_document_text_list,relevant_document_img_list
 
def load_data():
    # As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
    # about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder
    wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'
    if not os.path.exists(wikipedia_filepath):
        util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

    passages = []
    with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())

            #Add all paragraphs
            #passages.extend(data['paragraphs'])

            #Only add the first paragraph
            passages.append(data['paragraphs'][0])

    print("Passages:", len(passages))
    return passages

if __name__ == "__main__":
    img_dir="~/workspace/data/unsplash/photos"
    download_image(img_dir)