import os
 
from retrieval.utils.news import News
import pandas as pd
import nltk 
def get_relevant_document_dir(splited_dir):
    from pathlib import Path
    path = Path(splited_dir)
    whole_dataset_dir=path.parent.absolute()
    return whole_dataset_dir
 
def read_news(data_path,content="all",skip_empty_text_evidence=False,paragraph_sent_num=None): #text_evidence, all, txt, img
    news_dict={}
    news_dict=read_claim_and_evidence(data_path,news_dict,skip_empty_text_evidence,paragraph_sent_num)
    if content in ["all","txt"]:
        news_dict,relevant_document_text_list=read_relevant_document(data_path,news_dict)
    else:
        relevant_document_text_list=None
    if content in ["all","img","evidence"]:
        news_dict,relevant_document_img_list=read_image(data_path,news_dict)
    else:
        relevant_document_img_list=None
    return news_dict,relevant_document_text_list,relevant_document_img_list

def read_image(data_path,news_dict,content="all"):
    relevant_document_img_list=[]
    image_corpus=os.path.join(data_path,"images")
    img_list=os.listdir(image_corpus)
    for img_name in img_list:
        prefix=img_name[:17]
        ids=prefix.split("-")
        claim_id= int(ids[0])  
        relevant_document_id= ids[1] 
        if claim_id in news_dict:
            example=news_dict[claim_id]
        else:
            if content!="img":
                continue
            else:
                example=News(claim_id,None,None,None,None,None,None)   
        if relevant_document_id=="proof":
            example.add_img_evidence(img_name)
        elif content in ["all","img"]:
            relevant_document_id=int(relevant_document_id) 
            example.add_relevant_doc_img(img_name,relevant_document_id)
            relevant_document_img_list.append(img_name)
        news_dict[claim_id]=example
    return news_dict,relevant_document_img_list





def load_corpus_df(data_folder,corpus_name='Corpus3_sentence_level.csv'):
    relevant_document_dir=get_relevant_document_dir(data_folder)
    if corpus_name=="Corpus3_sentence_level.csv":
        collection_filepath = os.path.join(relevant_document_dir, "supplementary", corpus_name)
    else:
        collection_filepath = os.path.join(relevant_document_dir, corpus_name)
    df_result = pd.read_csv(collection_filepath ,encoding="utf8")
    
    # collection_filepath = os.path.join(relevant_document_dir,"train", corpus_name)
    # df_train = pd.read_csv(collection_filepath ,encoding="utf8")
    # collection_filepath = os.path.join(relevant_document_dir,"test", corpus_name)
    # df_test = pd.read_csv(collection_filepath ,encoding="utf8")
    # frames = [df_train, df_val,df_test] 
    # frames = [ df_val ]#TODO
    
    # df_result = pd.concat(frames)
    return df_result

def read_relevant_document(data_path,news_dict):
    df_relevant_doc=load_corpus_df(data_path,corpus_name="Corpus3.csv")
    # relevant_doc_corpus=os.path.join(data_path,"Corpus3.csv")
    relevant_document_text_list=[]
    # df_relevant_doc = pd.read_csv(relevant_doc_corpus ,encoding="utf8")
    for _,row in df_relevant_doc.iterrows():
        claim_id=row["claim_id"]
        relevant_document_id=row["relevant_document_id"]
        relevant_document_text =row["Origin Document"]
        if claim_id in news_dict:
            example=news_dict[claim_id]
            
            example.add_relevant_doc(relevant_document_text,relevant_document_id)
            relevant_document_text_list.append(relevant_document_text)
        else:
            example=News(claim_id,None,None,None,None,None,None)   
            example.add_relevant_doc(relevant_document_text,relevant_document_id)
            relevant_document_text_list.append(relevant_document_text)
        news_dict[claim_id]=example
    return news_dict,relevant_document_text_list

def read_claim_and_evidence(data_path,news_dict,skip_empty_text_evidence,paragraph_sent_num):
    evidence_corpus=os.path.join(data_path,"Corpus2.csv")
    df_news = pd.read_csv(evidence_corpus ,encoding="utf8")
    for _,row in df_news.iterrows():
        claim_id=row["claim_id"]
         
        if claim_id in news_dict:
            example=news_dict[claim_id]
            evidence=row["Evidence"]
            if not pd.isna(evidence) and len(evidence)>0:
                evidence=evidence.replace("<p>","")
                evidence=evidence.replace("</p>","")
                if paragraph_sent_num==1:
                    evidence_sent_list=nltk.tokenize.sent_tokenize(evidence)
                    example.evidence_dict["txt_list"].extend(evidence_sent_list)
                else:
                    example.add_text_evidence(evidence)
        else:
            if skip_empty_text_evidence and pd.isna(row["Evidence"]):
                continue
            else:
                example=News(claim_id,row["Snopes URL"],"",row['Claim'],row['cleaned_truthfulness'],row['Origin'],row["ruling_outline"])
                evidence=row["Evidence"]
                if not pd.isna(evidence) and len(evidence)>0:
                    evidence=evidence.replace("<p>","")
                    evidence=evidence.replace("</p>","")
                    if paragraph_sent_num==1:
                        evidence_sent_list=nltk.tokenize.sent_tokenize(evidence)
                        example.evidence_dict["txt_list"].extend(evidence_sent_list)
                    else:
                        example.add_text_evidence(evidence)
                news_dict[claim_id]=example
    return news_dict


if __name__ == '__main__':

    read_news("/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mode3_latest")