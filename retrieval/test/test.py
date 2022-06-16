from transformers import pipeline 
import torch 
import numpy as np
from nltk import tokenize 
from retrieval.search.image_search import ImageSearcher
from retrieval.search.lexical_search import LexicalSearcher
from retrieval.search.semantic_search import SemanticSearcher
from torch.utils.data import Dataset 
 
from torch.utils.data import DataLoader
from  retrieval.utils.data_util import *
import os 
from PIL import Image
import cv2

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
from  retrieval.utils.metrics import ImageScorer, TextScorer 
def test1():
    print(pipeline('sentiment-analysis')('we love you'))
    if not torch.cuda.is_available():
        print("Warning: No GPU found. Please add GPU to your notebook")
    else:
        print(f"cuda {torch.cuda.is_available()}")


# test1()


def test2():
    a = [0.6,0.9,0.65,0.7,0.02,0.8,0.653,0.5,0.95,0.1]
    print(a)
    a_partition=np.argpartition(a,(0,5))
    print(a_partition)
    a_partition2=np.argpartition(a,-5)
    print(a_partition2)
    a_sort=np.argsort(a)
    print(a_sort)


def test_text_search():
    top_k = 32                          #Number of passages we want to retrieve with the bi-encoder
    passages=load_data()
    query = "What is the capital of the United States?"
    
    searcher=SemanticSearcher()
    searcher.encode_corpus(passages)
    semantic_results=searcher.search(query,top_k)

    lexical_searcher=LexicalSearcher()
    lexical_searcher.encode_corpus(passages)
    lexical_results=lexical_searcher.search(query,top_k)



def test_image_search():
    top_k = 3                        
    query = "Two dogs playing in the snow"
    img_dir="/home/menglong/workspace/data/unsplash/photos"
    input_dir= "/home/menglong/workspace/code/misinformation_detection/input"
    emb_dir=input_dir+"/embeddings"
    
    searcher=ImageSearcher()
    searcher.encode_corpus(img_dir,None,emb_dir,True)    
    results=searcher.search(query,top_k)

def test5():
    # image = Image.open(os.path.join("~/workspace/data/unsplash/photos", "FAcSe7SjDUU.jpg"))
    # image = Image.open(os.path.join("input/photos", "FAcSe7SjDUU.jpg"))
    image = Image.open(os.path.join("/home/menglong/workspace/data/unsplash/photos", "FAcSe7SjDUU.jpg"))
    image.show()


 
def check_img():
    data_path="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mode3_latest"
    image_corpus=os.path.join(data_path,"images")
    img_names=os.listdir(image_corpus)
    
    for filepath in  img_names:
        

        im = cv2.imread(os.path.join(image_corpus,filepath))
        
        if im is None :
            print(filepath)
            
            os.remove(os.path.join(image_corpus,filepath))
        elif im.shape[2]!=3 :
            print(f"{im.shape},  {filepath}")
   
            os.remove(os.path.join(image_corpus,filepath))


 
def remove_img():
    data_path="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mode3_latest"
    image_corpus=os.path.join(data_path,"images")
    img_names=os.listdir(image_corpus)
    
    for img_name in  img_names:
        prefix=img_name[:13]
        ids=prefix.split("-")
        claim_id= int(ids[0]) #+1 #TODO diff by 1
        relevant_document_id=int(ids[1])#+1
        if claim_id>=597:
            os.remove(os.path.join(image_corpus,img_name))





def test_text_scorer1():
    
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Two lists of sentences
    sentences1 = ['The cat sits outside',
                'A man is playing guitar',
                'The new movie is awesome']

    sentences2 = ['The dog plays in the garden',
                'A man is playing TV',
                'The new movie is so great']

    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    print(cosine_scores)


def test7():
    df=pd.read_csv("output/evidence/train/evidences.csv")  
    df.head()
    # self.df.insert(-1, column_name, column_value_list)
    # df[column_name]=column_value_list
    df = df.drop('Unnamed: 0', 1)
    df = df.drop('Unnamed: 0.1', 1)
    df.head()
    df.to_csv("output/evidence/train/evidences1.csv")

def test10():
    import numpy as np
    x = np.random.randn(5)
    evidence=""
    for i in range(len(x)):
        evidence+=str(x[i])+" "
    print(evidence)
    
        
def test9():
    dataset1=load_dataset("xsum")
    print(dataset1)

def test11(data_path):
    dataset=SnopesDataset( data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
    length=len(dataloader)
    for iters in range(length):
        claim,text_evidence_list,truthfulness,claim_id,img_evidence_list,ruling_outline,ruling_article =dataloader.dataset[iters]
         
        if iters%1000==0:
            print(f"{iters},{length}")
        if len(text_evidence_list)>0  : 
            for document_evidence in text_evidence_list:
                try:
                    tokenize.sent_tokenize(document_evidence)
                except Exception as e:
                    print(f"{document_evidence},{claim_id},{iters}")
    
    
    # df = pd.read_csv(os.path.join(data_path,"Corpus2.csv") ,encoding="utf8")
    
    # for idx,row in df_evidence.iterrows():
    #     evidence =row["Evidence"]#
        # try:
        #     tokenize.sent_tokenize(evidence)
        # except Exception as e:
        #     print(f"{evidence},{row['claim_id']},{idx}")
 
if __name__ == "__main__":
    
    test11("/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/train")  