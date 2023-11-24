

from retrieval.utils.news import News
from util.preprocess import merge_evidence

import os
import pandas as pd

from util.read_example import read_image
 
def drop_invalid_data_for_verification(df):
    return df
    
    
def preprocess_for_verification_one_subset(data_path):
    out_name="Corpus2_for_verification.csv"
    merge_evidence( data_path,out_name,drop_invalid_data_for_verification)
    clean_text_evidence( data_path,out_name) 
    add_image_evidence(data_path,out_name)

 
def clean_text_evidence(data_path,out_name):
    corpus=os.path.join(data_path,out_name)
    evidence_list=[]
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    for i,row in evidence_df.iterrows():
        evidence=row["Evidence"]
        ruling_article=row["Origin"]
        ruling_outline=row["ruling_outline"]
        if pd.isna(evidence) or len(evidence)<=0:
            if pd.isna(ruling_outline) or len(ruling_outline)<=0:
                evidence=ruling_article #use ruling article to be evidence
            else:
                evidence=ruling_outline
        evidence_list.append(evidence)
    evidence_df =evidence_df.drop(columns=[ 'Evidence'])
    evidence_df.insert(13, "Evidence",evidence_list )
    evidence_df.to_csv(corpus,index=False)
 


 
def add_image_evidence(data_path,out_name):
    news_dict={}
    news_dict,_=read_image(data_path,news_dict,content="img")
    corpus=os.path.join(data_path,out_name)
    img_evidence_str_list=[]
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    for i,row in evidence_df.iterrows():
        claim_id=row["claim_id"]
        if claim_id in news_dict.keys():
            cur_img_evidence_list=news_dict[claim_id].get_img_evidence_list()
            cur_img_evidence_str=";".join(cur_img_evidence_list) 
        else:
            cur_img_evidence_str=""
        img_evidence_str_list.append(cur_img_evidence_str)
    evidence_df.insert(14, "img_evidences",img_evidence_str_list )
    evidence_df.to_csv(corpus,index=False)
     


def preprocess_for_verification(data_path):
    preprocess_for_verification_one_subset( data_path+ "/train")
    print("finish one")
    preprocess_for_verification_one_subset( data_path+"/val")
    print("finish one")
    preprocess_for_verification_one_subset(  data_path+"/test")
        
        
        
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2")
    args = parser.parse_args()
    return args
 



if __name__ == '__main__':
    args = parser_args()
    preprocess_for_verification_one_subset( args.data_path+ "/train")